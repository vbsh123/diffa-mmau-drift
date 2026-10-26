"""Read-only answer hidden-state drift, retaining token identity controls."""
from contextlib import AbstractContextManager
import json
from pathlib import Path

import numpy as np
import torch

from .protocol import ANSWER_TOKENS
from .storage import write_json

ANSWER_GROUPS=('answer_content_all','answer_content_masked_stable',
               'answer_content_decoded_stable','answer_content_changed')


class TransitionObservers:
    def __init__(self,*observers):
        self.observers=[o for o in observers if o is not None]

    def record_transition(self,x,x0,selected):
        for observer in self.observers:
            observer.record_transition(x,x0,selected)


def token_cosine_distance(current,reference):
    a,b=current.float(),reference.float()
    an,bn=a.norm(dim=-1),b.norm(dim=-1)
    cosine=((a*b).sum(-1)/(an*bn).clamp_min(1e-20)).clamp(-1,1)
    cosine=torch.where((an==0)&(bn==0),1.,cosine)
    if not torch.isfinite(cosine).all():
        raise ValueError('Nonfinite answer hidden-state drift')
    return 1-cosine


class AnswerDriftRecorder(AbstractContextManager):
    """Record all answer positions, then restrict summaries to the final visible span.

    Per-position cosine scalars are buffered until EOS/content length is known.
    Only previous/initial BF16 states are held, never a full hidden trajectory.
    """
    def __init__(self,llm,prepared,directory):
        self.llm,self.prepared,self.directory=llm,prepared,Path(directory)
        self.blocks=list(llm.model.transformer.blocks)
        self.ids=torch.full((ANSWER_TOKENS,),prepared.mask_id,dtype=torch.long)
        self.previous,self.initial={},{}
        self.step=-1
        self.rows=[]
        self.handles=[]
        self.seen=set()
        self.transitions=0

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        try:
            self.handles.append(self.llm.register_forward_pre_hook(self.begin,with_kwargs=True))
            for layer,block in enumerate(self.blocks):
                def hook(module,args,output,layer=layer):
                    self.observe(layer,output[0])
                self.handles.append(block.register_forward_hook(hook))
        except BaseException:
            self.__exit__(None,None,None)
            raise
        return self

    def begin(self,module,args,kwargs):
        if self.step>=0 and (len(self.seen)!=len(self.blocks) or self.transitions!=self.step+1):
            raise ValueError('Missing answer layer observations or official decision transition')
        self.step+=1
        self.seen=set()
        embeds=kwargs.get('inputs_embeds')
        if embeds is None or embeds.shape[1]!=self.prepared.prefix_length+ANSWER_TOKENS or kwargs.get('use_cache'):
            raise ValueError('Unexpected dense answer-drift canvas')

    def observe(self,layer,hidden):
        if layer in self.seen:
            raise ValueError('Answer layer observed twice')
        self.seen.add(layer)
        state=hidden[0,self.prepared.prefix_length:].detach().cpu().clone()
        if state.shape[0]!=ANSWER_TOKENS or not torch.isfinite(state).all():
            raise ValueError('Unexpected/nonfinite answer states')
        ids=self.ids.clone()
        if layer not in self.initial:
            self.initial[layer]=(state,ids)
        for reference,bank in (('previous',self.previous),('initial',self.initial)):
            if layer not in bank:
                continue
            old,old_ids=bank[layer]
            same=ids==old_ids
            groups=(torch.ones_like(same),same&(ids==self.prepared.mask_id),
                    same&(ids!=self.prepared.mask_id),~same)
            self.rows.append((self.step,layer,reference,
                              token_cosine_distance(state,old).numpy(),
                              [mask.numpy() for mask in groups]))
        self.previous[layer]=(state,ids)

    def record_transition(self,x,x0,selected):
        current=x[0,self.prepared.prefix_length:].detach().cpu()
        if not torch.equal(current,self.ids):
            raise ValueError('Answer-drift IDs diverged from official FPD')
        chosen=selected[0,self.prepared.prefix_length:].detach().cpu()
        proposed=x0[0,self.prepared.prefix_length:].detach().cpu()
        self.ids=current.clone()
        self.ids[chosen]=proposed[chosen]
        self.transitions+=1

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self,visible_tokens):
        if not 0<=visible_tokens<=ANSWER_TOKENS or len(self.seen)!=len(self.blocks) or self.transitions!=self.step+1:
            raise ValueError('Incomplete answer drift or invalid final visible span')
        with (self.directory/'answer_drift.jsonl').open('w') as stream:
            for step,layer,reference,distance,masks in self.rows:
                if step==0:
                    continue  # No drift comparison at initialization.
                for group,mask in zip(ANSWER_GROUPS,masks):
                    selected=distance[:visible_tokens][mask[:visible_tokens]]
                    if not len(selected):
                        continue  # Absent populations are not zero-filled.
                    row=dict(step=step,layer=layer,reference=reference,group=group,count=len(selected),
                             metrics=dict(mean=float(selected.mean()),median=float(np.median(selected)),
                                          p90=float(np.quantile(selected,.9)),max=float(selected.max())))
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(self.directory/'answer_measurement.json',dict(
            protocol='answer-hidden-drift-visible-span-v1',steps=self.step+1,layers=list(range(len(self.blocks))),
            groups=ANSWER_GROUPS,visible_tokens=visible_tokens,canvas_tokens=ANSWER_TOKENS,
            span='Retrospective final visible answer span, excludes first EOS and later slots; includes reasoning',
            hidden_state='Post-transformer-block output at the same answer position',
            primary='Mean per-position FP32 cosine distance',references=['previous','initial'],
            identity_controls='Stable masked/decoded compare identical token IDs in current and reference forwards',
            changed_group='Different token IDs; change includes input-token replacement and contextual change',
            empty_groups='Omitted, never padded with zero',initial_decoded='Absent when reference forward0 is all MASK',
            note='Population membership depends on the reference; curves are descriptive and not cache error.'))
        self.rows.clear()
        self.previous.clear()
        self.initial.clear()
