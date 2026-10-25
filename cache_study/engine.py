"""Prefix K/V and block-output reuse in selected layers, with dense warmup."""
from contextlib import AbstractContextManager

import torch


class PrefixCache(AbstractContextManager):
    def __init__(self, llm, prepared, intervals=(16,16,16), layers=None, start_forward=0):
        self.blocks = list(llm.model.transformer.blocks)
        if llm.training or len(self.blocks)!=llm.config.n_layers or len(intervals)!=3 or min(intervals)<1:
            raise ValueError('Requires eval-mode ungrouped LLaDA and three positive intervals')
        self.groups = [prepared.groups()[g] for g in ('acoustic','text','semantic')]
        self.llm = llm
        self.layers = set(range(len(self.blocks))) if layers is None else set(layers)
        if not self.layers or not self.layers <= set(range(len(self.blocks))) or start_forward < 0:
            raise ValueError('Invalid cache layers or start forward')
        self.start_forward = start_forward
        self.anchor = max(0, start_forward-1)
        self.handle = None
        if any(not g for g in self.groups) or sum(map(len,self.groups))!=prepared.prefix_length:
            raise ValueError('Expected nonempty acoustic/prompt/semantic groups partitioning the prefix')
        positions = [p for group in self.groups for p in group]
        if sorted(positions) != list(range(prepared.prefix_length)):
            raise ValueError('Prefix groups overlap or have gaps')
        self.intervals = tuple(intervals)
        self.step, self.length = -1, None
        self.banks, self.patterns, self.originals, self.events = {}, {}, [], []
        self.pattern_ids, self.active_ids_by_layer = {}, {}

    def __enter__(self):
        try:
            self.handle = self.llm.register_forward_pre_hook(self.begin, with_kwargs=True)
            for layer,block in enumerate(self.blocks):
                if layer not in self.layers:
                    continue
                if (type(block).__name__!='LLaDALlamaBlock' or block._activation_checkpoint_fn is not None
                        or not block.config.rope or block.config.n_heads!=block.config.effective_n_kv_heads):
                    raise ValueError('Requires pinned equal-head rotary LLaDALlamaBlock')
                original = block.forward
                self.originals.append((block,'forward' in block.__dict__,block.__dict__.get('forward')))

                def forward(x,attention_bias=None,layer_past=None,use_cache=False,replace_position=None,
                            _layer=layer,_block=block,_original=original):
                    if x.shape[0]!=1 or attention_bias is not None or layer_past is not None or use_cache or replace_position is not None:
                        raise ValueError('Requires batch-one bidirectional no-past inference')
                    active=self.active
                    self.active_ids_by_layer[_layer]=self.pattern_ids[self.events[-1]['refresh']]
                    if self.step < self.anchor:
                        return _original(x)
                    if len(active)==self.length:
                        # Preserve the pinned upstream path for full refreshes.
                        attention=_block._scaled_dot_product_attention
                        def capture(q,k,v,**kwargs):
                            self.banks[_layer]=[k.detach().clone(),v.detach().clone(),None]
                            return attention(q,k,v,**kwargs)
                        _block._scaled_dot_product_attention=capture
                        try:
                            output=_original(x)
                        finally:
                            _block._scaled_dot_product_attention=attention
                        self.banks[_layer][2]=output[0].detach().clone()
                        return output
                    y=x.index_select(1,active)
                    norm=_block.attn_norm(y)
                    q,k,v=_block.q_proj(norm),_block.k_proj(norm),_block.v_proj(norm)
                    dtype=k.dtype
                    if _block.q_norm is not None and _block.k_norm is not None:
                        q,k=_block.q_norm(q).to(dtype),_block.k_norm(k).to(dtype)
                    batch,rows,width=q.shape
                    heads=_block.config.n_heads
                    q=q.view(batch,rows,heads,width//heads).transpose(1,2)
                    k=k.view(batch,rows,heads,width//heads).transpose(1,2)
                    v=v.view(batch,rows,heads,width//heads).transpose(1,2)
                    qr,kr=(q.float(),k.float()) if _block.config.rope_full_precision else (q,k)
                    rotary=_block.rotary_emb
                    with torch.autocast(q.device.type,enabled=False):
                        sin,cos=rotary.get_rotary_embedding(self.length,q.device)
                        sin=sin.type_as(qr).index_select(-2,active)
                        cos=cos.type_as(qr).index_select(-2,active)
                        qr=rotary.apply_rotary_pos_emb(sin,cos,qr)
                        kr=rotary.apply_rotary_pos_emb(sin,cos,kr)
                    full_k,full_v,hidden=self.banks[_layer]
                    full_k.index_copy_(-2,active,kr.type_as(k))
                    full_v.index_copy_(-2,active,v)
                    attended=_block._scaled_dot_product_attention(qr.type_as(q),full_k,full_v,
                        attn_mask=None,dropout_p=0.,is_causal=False)
                    attended=attended.transpose(1,2).contiguous().view(batch,rows,width)
                    y=y+_block.dropout(_block.attn_out(attended))
                    norm=_block.ff_norm(y)
                    y=y+_block.dropout(_block.ff_out(_block.act(_block.ff_proj(norm))*_block.up_proj(norm)))
                    hidden.index_copy_(1,active,y)
                    return hidden,None
                block.forward=forward
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def begin(self, module, args, kwargs):
        x = kwargs.get('inputs_embeds')
        if x is None or x.shape[0] != 1 or kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
            raise ValueError('Requires batch-one dense-canvas FPD input embeddings')
        self.step += 1
        if self.length is None:
            self.length = x.shape[1]
        if x.shape[1] != self.length:
            raise ValueError('FPD canvas length changed')
        due = tuple(self.step <= self.anchor or (self.step-self.anchor)%r == 0 for r in self.intervals)
        if due not in self.patterns:
            inactive = {p for group,refresh in zip(self.groups,due) if not refresh for p in group}
            selected = [p for p in range(self.length) if p not in inactive]
            self.pattern_ids[due] = tuple(selected)
            self.patterns[due] = torch.tensor(selected,device=x.device,dtype=torch.long)
        self.active = self.patterns[due]
        self.events.append(dict(step=self.step,refresh=due,computed_rows=len(self.active)))

    def __exit__(self,*args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for block,existed,original in reversed(self.originals):
            if existed:
                block.forward=original
            else:
                delattr(block,'forward')
        self.originals.clear()

    def accounting(self):
        selected_used = sum(e['computed_rows'] for e in self.events)
        ordinary_used = len(self.events)*(self.length or 0)
        used = selected_used*len(self.layers)+ordinary_used*(len(self.blocks)-len(self.layers))
        dense=len(self.events)*(self.length or 0)*len(self.blocks)
        return dict(forwards=len(self.events),computed_token_layers=used,dense_token_layers=dense,
                    cache_layers=sorted(self.layers), dense_warmup_forwards=self.start_forward,
                    actual_reuse_forwards=sum(e['computed_rows'] < self.length for e in self.events),
                    per_layer_computed_rows={str(layer):selected_used if layer in self.layers else ordinary_used
                                             for layer in range(len(self.blocks))},
                    skipped_token_update_fraction=1-used/max(1,dense),
                    refresh_counts={g:sum(e['refresh'][i] for e in self.events)
                                    for i,g in enumerate(('acoustic','text','semantic'))},
                    note='Only fixed-prefix QKV/query/MLP rows are skipped. All KV columns and answer rows remain; '
                         'original positions, full embedding, final norm and vocabulary head are retained.')
