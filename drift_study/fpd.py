"""Call the unchanged official FPD generator; trace its decisions without replacing them."""
from contextlib import contextmanager
import json

import torch

from .protocol import ANSWER_TOKENS, BLOCK_TOKENS, FACTOR


@contextmanager
def decision_trace(upstream, prepared, max_forwards=1024, observer=None):
    """Synchronous batch-one observation wrapper; return original tensors untouched."""
    original = upstream.get_transfer_index_dynamic
    rows = []

    def observe(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
        if len(rows) >= max_forwards:
            raise RuntimeError('Official FPD exceeded watchdog; abort without forcing commits')
        result = original(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor)
        x0, selected = result
        if observer is not None:
            observer.record_transition(x,x0,selected)
        start = prepared.prefix_length
        positions = selected[0].nonzero(as_tuple=True)[0].detach().cpu().tolist()
        if any(p < start for p in positions):
            raise ValueError('FPD selected a fixed-prefix position')
        answer_positions = [p-start for p in positions]
        chosen = x0[0, positions].detach().cpu().tolist()
        remaining = (x[0, start:] == prepared.mask_id)
        rows.append(dict(step=len(rows), masked_before=int(remaining.sum()),
            eligible_masks=int(mask_index.sum()), commits=len(positions),
            answer_positions=answer_positions, token_ids=chosen,
            block=min(answer_positions)//BLOCK_TOKENS if answer_positions else None))
        return result

    upstream.get_transfer_index_dynamic = observe
    try:
        yield rows
    finally:
        upstream.get_transfer_index_dynamic = original


@torch.no_grad()
def generate(model, prepared, trace=True, observer=None):
    from src.fast_dllm import diffa_generate as upstream
    kwargs = dict(audio_info_lengths=torch.tensor([prepared.insertion], device=prepared.prompt.device),
        speech_embeds=prepared.speech, steps=ANSWER_TOKENS, gen_length=ANSWER_TOKENS,
        block_length=BLOCK_TOKENS, temperature=0., remasking='low_confidence',
        mask_id=prepared.mask_id, threshold=None, factor=FACTOR)
    if trace:
        with decision_trace(upstream, prepared, observer=observer) as rows:
            output, nfe = upstream.generate(model, prepared.prompt, **kwargs)
        if len(rows) != nfe:
            raise ValueError('Trace does not cover every FPD forward')
    else:
        if observer is not None:
            raise ValueError('Attention state observation requires the official decision trace')
        output, nfe = upstream.generate(model, prepared.prompt, **kwargs)
        rows = []
    return output, int(nfe), rows


def output_record(output, prepared, tokenizer):
    ids = output[0, prepared.prompt.shape[1]:].detach().cpu().tolist()
    eos = tokenizer.eos_token_id
    has_eos = eos in ids
    visible = ids[:ids.index(eos)] if has_eos else ids
    unresolved = visible.count(prepared.mask_id)
    return dict(canvas_ids=ids, visible_ids=visible, text=tokenizer.decode(visible, skip_special_tokens=True).strip(),
                eos_found=has_eos, capacity_exhausted=not has_eos,
                unresolved_visible_masks=unresolved, visible_tokens=len(visible))


def save_trace(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
