"""Read-only block-output observations; CPU references and bounded matrix scratch."""
from contextlib import AbstractContextManager
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .protocol import ANSWER_TOKENS, GROUPS, PAIRWISE_LAYERS
from .storage import write_json


def distances(current, previous):
    """Mean per-position cosine and cosine of flattened group, both in FP32."""
    a, b = current.float(), previous.float()
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0:
        raise ValueError('Expected matching nonempty token x feature tensors')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite hidden state')
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    similarity = ((a*b).sum(-1)/(an*bn).clamp_min(1e-20)).clamp(-1, 1)
    similarity = torch.where((an == 0) & (bn == 0), 1., similarity)
    flat_denom = a.norm()*b.norm()
    flat = ((a*b).sum()/flat_denom.clamp_min(1e-20)).clamp(-1, 1)
    if flat_denom == 0:
        flat = torch.tensor(1. if a.norm() == 0 and b.norm() == 0 else 0.)
    difference = 1-similarity
    return dict(mean=float(difference.mean()), median=float(torch.quantile(difference, .5)),
                p90=float(torch.quantile(difference, .9)), max=float(difference.max()),
                flattened=float(1-flat), zero_norm_tokens=int(((an == 0) | (bn == 0)).sum()))


def pairwise_matrices(paths, positions, chunk_tokens=8):
    """Exact per-token and flattened cosine matrices, reading only small token chunks."""
    arrays = [np.load(path, mmap_mode='r') for path in paths]
    count = len(arrays)
    if not count or not positions:
        raise ValueError('Empty pairwise input')
    token_cosine = torch.zeros(count, count, dtype=torch.float32)
    flat_dot = torch.zeros_like(token_cosine)
    for start in range(0, len(positions), chunk_tokens):
        index = positions[start:start+chunk_tokens]
        block = torch.from_numpy(np.stack([array[index].astype(np.float32) for array in arrays]))
        norms = block.norm(dim=-1, keepdim=True)
        normalized = block/norms.clamp_min(1e-20)
        normalized = normalized.reshape(count, -1)
        contribution = normalized @ normalized.T
        zeros = (norms.squeeze(-1) == 0).float()
        contribution += zeros @ zeros.T
        token_cosine += contribution
        raw = block.reshape(count, -1)
        flat_dot += raw @ raw.T
    token_cosine = (token_cosine/len(positions)).clamp(-1, 1)
    norm = flat_dot.diag().clamp_min(0).sqrt()
    denominator = norm[:, None]*norm[None, :]
    flat_cosine = (flat_dot/denominator.clamp_min(1e-20)).clamp(-1, 1)
    flat_cosine = torch.where((norm[:, None] == 0) & (norm[None, :] == 0), 1., flat_cosine)
    return token_cosine.numpy(), flat_cosine.numpy()


class HiddenRecorder(AbstractContextManager):
    def __init__(self, llm, prepared, directory, pairwise=False):
        self.llm, self.prepared = llm, prepared
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.groups = prepared.groups()
        self.blocks = list(llm.model.transformer.blocks)
        if len(self.blocks) != llm.config.n_layers:
            raise ValueError('Expected individual, ungrouped transformer blocks')
        self.layers = len(self.blocks)
        self.pairwise_layers = [i for i in PAIRWISE_LAYERS if i < self.layers] if pairwise else []
        self.scratch = self.directory/'matrix_scratch'
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        self.previous, self.initial = {}, {}
        self.handles, self.seen = [], set()
        self.step = -1
        self.stream = None

    def __enter__(self):
        self.stream = (self.directory/'drift.jsonl').open('w')
        try:
            self.handles.append(self.llm.register_forward_pre_hook(self.begin, with_kwargs=True))
            for layer, block in enumerate(self.blocks):
                def hook(module, inputs, output, layer=layer):
                    self.observe(layer, output[0])
                self.handles.append(block.register_forward_hook(hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def begin(self, module, inputs, kwargs):
        if self.step >= 0 and len(self.seen) != self.layers:
            raise ValueError('Incomplete layer coverage')
        if kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
            raise ValueError('Drift requires dense uncached forwards')
        embeds = kwargs.get('inputs_embeds')
        if embeds is None or embeds.shape[0] != 1 or embeds.shape[1] != self.prepared.prefix_length+ANSWER_TOKENS:
            raise ValueError('Unexpected official FPD canvas shape')
        self.step += 1
        self.seen = set()

    def observe(self, layer, hidden):
        if layer in self.seen:
            raise ValueError('Layer observed twice')
        self.seen.add(layer)
        # clone owns CPU storage and never aliases the running model.
        current = hidden[0, :self.prepared.prefix_length].detach().cpu().clone()
        if not torch.isfinite(current).all():
            raise ValueError('Nonfinite prefix state')
        if layer in self.pairwise_layers:
            destination = self.scratch/f'layer_{layer:02d}'
            destination.mkdir(parents=True, exist_ok=True)
            # BF16 values are exactly representable in FP32; preserve them losslessly.
            # FP16 scratch would overflow large BF16 outliers, so use raw BF16 words.
            if current.dtype != torch.bfloat16:
                raise ValueError('Official FPD matrix snapshots must be BF16')
            np.save(destination/f'{self.step:04d}.npy', current.view(torch.uint16).numpy())
        if layer not in self.initial:
            self.initial[layer] = current
        for reference, bank in (('previous', self.previous), ('initial', self.initial)):
            if layer not in bank:
                continue
            for group in GROUPS:
                index = self.groups[group]
                row = dict(step=self.step, layer=layer, group=group, reference=reference,
                           count=len(index), metrics=distances(current[index], bank[layer][index]))
                self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.previous[layer] = current

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.stream:
            self.stream.close()

    def finalize(self):
        if len(self.seen) != self.layers:
            raise ValueError('Final forward missing measured layers')
        if self.pairwise_layers:
            matrices = {}
            for layer in self.pairwise_layers:
                paths = sorted((self.scratch/f'layer_{layer:02d}').glob('*.npy'))
                if len(paths) != self.step+1:
                    raise ValueError('Missing matrix snapshot')
                # Temporary converted views per layer, removed once its matrices exist.
                converted = []
                for path in paths:
                    words = np.load(path)
                    values = torch.from_numpy(words).view(torch.bfloat16).float().numpy()
                    target = path.with_suffix('.float.npy')
                    np.save(target, values)
                    converted.append(target)
                for group in GROUPS:
                    token, flattened = pairwise_matrices(converted, self.groups[group])
                    matrices[f'{group}_layer{layer}_token_mean'] = token
                    matrices[f'{group}_layer{layer}_flattened'] = flattened
                for path in converted:
                    path.unlink()
            np.savez_compressed(self.directory/'pairwise.npz', **matrices)
            shutil.rmtree(self.scratch)
        write_json(self.directory/'measurement.json', dict(layers=list(range(self.layers)),
            layer_convention='zero-based transformer block outputs; embedding is not layer 0',
            steps=self.step+1, groups={g:len(p) for g,p in self.groups.items()},
            pairwise_layers=self.pairwise_layers, primary='mean per-position FP32 cosine distance',
            secondary='cosine distance of flattened group',
            references=['previous', 'initial'], input_text_includes_chat_delimiters=True))
