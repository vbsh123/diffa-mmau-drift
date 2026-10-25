"""Compact modality attention summaries from actual post-RoPE queries and keys."""
from contextlib import AbstractContextManager
import json
import math
from pathlib import Path

import torch

from .protocol import ANSWER_TOKENS
from .storage import write_json

ATTENTION_GROUPS=('acoustic','semantic','text','answer_masked','answer_decoded')


def group_attention_mass(q,k,indicators,query_chunk=128):
    """FP32 softmax reconstruction, bounded by heads x query_chunk x keys.

    Returns mean attention mass per query group and key group. Includes self
    attention and every head/key/query; no query subsampling or dense NxN archive.
    """
    if q.ndim!=4 or q.shape[0]!=1 or q.shape!=k.shape or query_chunk<1:
        raise ValueError('Expected batch-one full equal-head Q/K attention')
    if indicators.shape[1]!=q.shape[-2] or not torch.all(indicators.sum(0)==1):
        raise ValueError('Attention groups must partition the entire sequence')
    counts=indicators.sum(1)
    weights=indicators.float()
    key=k[0].float()
    accumulated=torch.zeros(len(counts),len(counts),device=q.device,dtype=torch.float32)
    for lo in range(0,q.shape[-2],query_chunk):
        hi=min(lo+query_chunk,q.shape[-2])
        scores=torch.matmul(q[0,:,lo:hi].float(),key.transpose(-1,-2))/math.sqrt(q.shape[-1])
        probabilities=scores.softmax(-1)
        per_query=probabilities.sum(0)@weights.T
        accumulated+=weights[:,lo:hi]@per_query
        del scores,probabilities,per_query
    mass=accumulated/(counts[:,None].clamp_min(1)*q.shape[1])
    relative=mass/(counts[None,:].clamp_min(1)/q.shape[-2])
    return mass,relative,counts


class AttentionRecorder(AbstractContextManager):
    def __init__(self,llm,prepared,directory,every=1,query_chunk=128):
        if every<1 or query_chunk<1:
            raise ValueError('Invalid attention observation settings')
        self.llm,self.prepared=llm,prepared
        self.directory=Path(directory)
        self.every,self.query_chunk=every,query_chunk
        self.blocks=list(llm.model.transformer.blocks)
        self.answer_ids=torch.full((ANSWER_TOKENS,),prepared.mask_id,
            device=prepared.prompt.device,dtype=torch.long)
        self.step=-1
        self.seen=set()
        self.originals=[]
        self.handle=None
        self.stream=None

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        self.stream=(self.directory/'attention.jsonl').open('w')
        try:
            self.handle=self.llm.register_forward_pre_hook(self.begin,with_kwargs=True)
            for layer,block in enumerate(self.blocks):
                original=block._scaled_dot_product_attention
                self.originals.append((block,'_scaled_dot_product_attention' in block.__dict__,
                                       block.__dict__.get('_scaled_dot_product_attention')))
                def observe(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,layer=layer,original=original):
                    if attn_mask is not None or dropout_p or is_causal:
                        raise ValueError('Expected unmasked bidirectional eval attention')
                    if self.step%self.every==0:
                        self.observe(layer,q,k)
                    # Native SDPA output is returned unchanged; reconstructed weights
                    # are observation only and never enter the model computation.
                    return original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
                block._scaled_dot_product_attention=observe
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def begin(self,module,args,kwargs):
        if self.step>=0 and self.step%self.every==0 and len(self.seen)!=len(self.blocks):
            raise ValueError('Missing observed attention layer')
        self.step+=1
        self.seen=set()
        length=self.prepared.prefix_length+ANSWER_TOKENS
        embeds=kwargs.get('inputs_embeds')
        if embeds is None or embeds.shape[1]!=length or kwargs.get('use_cache'):
            raise ValueError('Unexpected attention observation canvas')
        indicators=torch.zeros(len(ATTENTION_GROUPS),length,device=embeds.device,dtype=torch.bool)
        groups=self.prepared.groups()
        for i,group in enumerate(ATTENTION_GROUPS[:3]):
            indicators[i,groups[group]]=True
        masked=self.answer_ids==self.prepared.mask_id
        indicators[3,self.prepared.prefix_length:]=masked
        indicators[4,self.prepared.prefix_length:]=~masked
        self.indicators=indicators

    def observe(self,layer,q,k):
        if layer in self.seen:
            raise ValueError('Observed attention layer twice')
        self.seen.add(layer)
        mass,relative,counts=group_attention_mass(q,k,self.indicators,self.query_chunk)
        values,ratios,n=mass.detach().cpu().tolist(),relative.detach().cpu().tolist(),counts.cpu().tolist()
        row=dict(step=self.step,layer=layer,counts=dict(zip(ATTENTION_GROUPS,n)),mass={},relative_to_uniform={})
        for i,query in enumerate(ATTENTION_GROUPS):
            if n[i]:
                row['mass'][query]={key:values[i][j] for j,key in enumerate(ATTENTION_GROUPS) if n[j]}
                row['relative_to_uniform'][query]={key:ratios[i][j] for j,key in enumerate(ATTENTION_GROUPS) if n[j]}
        self.stream.write(json.dumps(row,allow_nan=False)+'\n')

    def record_transition(self,x,x0,selected):
        start=self.prepared.prefix_length
        current=x[0,start:]
        if not torch.equal(current,self.answer_ids):
            raise ValueError('Attention answer-state labels diverged from official FPD')
        self.answer_ids=current.detach().clone()
        active=selected[0,start:]
        self.answer_ids[active]=x0[0,start:][active]

    def __exit__(self,*exc):
        if self.handle is not None:
            self.handle.remove()
        for block,existed,value in reversed(self.originals):
            if existed:
                block._scaled_dot_product_attention=value
            else:
                delattr(block,'_scaled_dot_product_attention')
        self.originals.clear()
        if self.stream:
            self.stream.close()

    def finalize(self):
        if self.step%self.every==0 and len(self.seen)!=len(self.blocks):
            raise ValueError('Final attention observation has missing layers')
        write_json(self.directory/'attention_measurement.json',dict(
            groups=ATTENTION_GROUPS,layers=list(range(len(self.blocks))),
            every_forwards=self.every,query_chunk=self.query_chunk,
            metric='Mean softmax attention mass over all heads and queries within each query group',
            reconstruction='FP32 from native post-RoPE Q/K; native SDPA output unchanged',
            relative_to_uniform='mass / (key_group_count / full_sequence_length)',
            includes_self_attention=True,query_subsampling=False,full_matrices_saved=False))
