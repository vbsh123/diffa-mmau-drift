"""Run the 128-token official-FPD drift pilot on a frozen MMAU-Pro manifest."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import torch

from .audio_windows import INPUT_PROTOCOL
from .attention import AttentionRecorder
from .answer import AnswerDriftRecorder,TransitionObservers
from .checkout import COMMIT, DEFAULT_ROOT, verify_checkout
from .data import load_audio, read_manifest
from .fpd import generate, output_record, save_trace
from .metrics import HiddenRecorder
from .model import DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION, load_model, prepare
from .protocol import ANSWER_TOKENS, BLOCK_TOKENS, FACTOR, SEED, grade
from .storage import digest, source_digest, write_json

WHISPER = 'openai/whisper-large-v3'
WHISPER_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'


def verify_cache_reference(record,reference):
    if (reference['failed'] or reference['id']!=record['id']
            or reference['audio_sha256']!=record['audio_sha256']
            or reference['forward_count']!=record['forward_count']
            or reference['output']['canvas_ids']!=record['output']['canvas_ids']
            or reference['group_tokens']!=record['group_tokens']
            or reference['audio_chunks']!=record['audio_chunks']):
        raise ValueError('Measured drift run differs from the clean cache baseline; refusing mixed results')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/mmau50_long.json'))
    p.add_argument('--out', type=Path, default=Path('runs/mmau50_long_fpd128'))
    p.add_argument('--diffa-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--verify-samples', type=int, default=2)
    p.add_argument('--pairwise-domains', type=int, choices=(0, 1, 2, 3), default=3,
                   help='First example in each of the first N domains gets full step-pair matrices')
    p.add_argument('--cache-reference-run',type=Path,help='Require matching clean cache baseline on every measured question')
    p.add_argument('--no-answer-drift',action='store_true',help='Disable answer hidden-state observations')
    p.add_argument('--attention-every',type=int,default=1,help='Attention summaries every N forwards; 0 disables')
    p.add_argument('--attention-query-chunk',type=int,default=128)
    return p


def main():
    args = parser().parse_args()
    if args.samples < 1 or not 0 <= args.verify_samples <= args.samples:
        raise ValueError('Invalid sample or verification count')
    if args.attention_every<0 or args.attention_query_chunk<1:
        raise ValueError('Invalid attention observation settings')
    manifest = read_manifest(args.manifest)
    if manifest['settings'].get('input_protocol') != INPUT_PROTOCOL:
        raise ValueError('Prepare a fresh complete-clip long-audio manifest; old short cohorts are not reused')
    if args.samples > len(manifest['rows']):
        raise ValueError('Requested sample count exceeds frozen cohort')
    rows = manifest['rows'][:args.samples]
    verify_checkout(args.diffa_root)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run on a BF16-capable CUDA GPU (e.g. Vast A40 48GB)')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dict(protocol='mmau-pro-official-fpd-long-drift-v2', source_sha256=source_digest(),
        upstream_commit=COMMIT, manifest_sha256=digest(args.manifest),
        dataset=manifest['dataset'], dataset_revision=manifest['revision'],
        input_protocol=INPUT_PROTOCOL, audio_selection=manifest['settings'],
        attention=dict(every=args.attention_every,query_chunk=args.attention_query_chunk),
        answer_drift=dict(enabled=not args.no_answer_drift,protocol='answer-hidden-drift-visible-span-v1'),
        sample_ids=[r['id'] for r in rows], seed=args.seed,
        decoder=dict(name='official DIFFA-2 fast_dllm FPD', factor=FACTOR,
            max_answer_tokens=ANSWER_TOKENS, block_tokens=BLOCK_TOKENS, temperature=0.,
            fixed_commit_quota=False, fixed_confidence_threshold=False, temporal_cache=False),
        models=dict(diffa=[DIFFA_REPO, DIFFA_REVISION], llm=[LLM_REPO, LLM_REVISION],
                    whisper=[WHISPER, WHISPER_REVISION], lora_alpha=16),
        verify_samples=args.verify_samples, pairwise_domains=args.pairwise_domains,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            transformers=importlib.metadata.version('transformers'),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    if args.cache_reference_run:
        reference_config=json.loads((args.cache_reference_run/'config.json').read_text())
        if any(reference_config[key]!=config[key] for key in
               ('manifest_sha256','sample_ids','models','seed')):
            raise ValueError('Cache reference cohort/model/seed differs from drift run')
        reference_decoder=reference_config['decoder']
        if any(reference_decoder[key]!=config['decoder'][key] for key in
               ('max_answer_tokens','block_tokens','factor','temperature')):
            raise ValueError('Cache reference decoder differs from drift run')
        config['cache_reference']=dict(config_sha256=digest(args.cache_reference_run/'config.json'),
            baseline_sha256={str(i):digest(args.cache_reference_run/'samples'/f'{i:05d}'/'baseline.json')
                             for i in range(len(rows))})
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Source, environment, or settings changed; choose a fresh output directory')
    elif any(args.out.iterdir()):
        raise ValueError('Output directory has no matching config')
    else:
        write_json(config_path, config)
    frozen_manifest=args.out/'cohort.json'
    if frozen_manifest.exists():
        if digest(frozen_manifest)!=config['manifest_sha256']:
            raise ValueError('Frozen output cohort manifest was changed')
    else:
        shutil.copyfile(args.manifest,frozen_manifest)
    model, tokenizer, extractor, load_report = load_model(args.diffa_root,
        DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION, WHISPER, WHISPER_REVISION)
    write_json(args.out/'model_load.json', load_report)
    if tokenizer.eos_token_id != 126081 or model.llm_model.config.mask_token_id != 126336:
        raise ValueError('Unexpected tokenizer IDs for official DIFFA FPD')
    exemplars = []
    seen_domains = set()
    for row in rows:
        if row['domain'] not in seen_domains and len(exemplars) < args.pairwise_domains:
            exemplars.append(row['id'])
            seen_domains.add(row['domain'])
    completed_this_run, started = 0, time.monotonic()
    for index, row in enumerate(rows):
        directory = args.out/'samples'/f'{index:05d}'
        result_path = directory/'result.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved['id'] != row['id'] or saved['audio_sha256'] != row['audio_sha256']:
                raise ValueError('Existing sample identity mismatch')
            for name, expected in saved['artifact_sha256'].items():
                if digest(directory/name) != expected:
                    raise ValueError('Existing sample artifact changed or corrupted')
            if args.cache_reference_run:
                reference=json.loads((args.cache_reference_run/'samples'/f'{index:05d}'/'baseline.json').read_text())
                verify_cache_reference(saved,reference)
            print(f'Skip complete {index+1}/{len(rows)}: {row["id"]}', flush=True)
            continue
        if directory.exists():
            shutil.rmtree(directory)  # Only incomplete outputs owned by this matching run.
        directory.mkdir(parents=True)
        audio = load_audio(args.manifest.resolve().parent/row['audio'])
        prepared = prepare(model, tokenizer, extractor, audio, row['prompt'], ANSWER_TOKENS)
        expected = row['estimated_audio_tokens']
        actual = prepared.groups()
        if (len(actual['acoustic']),len(actual['semantic']),prepared.prompt.shape[1]) != (expected['acoustic'],expected['semantic'],row['text_tokens']):
            raise ValueError('Actual windowed audio/text token counts differ from cohort preflight')
        reference = None
        if index < args.verify_samples:
            print(f'Untouched FPD verification {index+1}/{args.verify_samples}', flush=True)
            reference = generate(model, prepared, trace=False)
        print(f'Measure {index+1}/{len(rows)} {row["domain"]} {row["audio_seconds"]:.2f}s '
              f'groups={ {g:len(p) for g,p in prepared.groups().items()} }', flush=True)
        torch.cuda.synchronize()
        began = time.monotonic()
        attention=None
        answer=None
        with ExitStack() as stack:
            recorder=stack.enter_context(HiddenRecorder(model.llm_model,prepared,directory,row['id'] in exemplars))
            if args.attention_every:
                attention=stack.enter_context(AttentionRecorder(model.llm_model,prepared,directory,
                    every=args.attention_every,query_chunk=args.attention_query_chunk))
            if not args.no_answer_drift:
                answer=stack.enter_context(AnswerDriftRecorder(model.llm_model,prepared,directory))
            output,nfe,trace=generate(model,prepared,observer=TransitionObservers(attention,answer))
        torch.cuda.synchronize()
        instrumented_seconds = time.monotonic()-began
        if nfe != recorder.step+1:
            raise ValueError('Hidden-state coverage differs from official NFE')
        verified = None
        if reference is not None:
            verified = bool(nfe == reference[1] and torch.equal(output, reference[0]))
            if not verified:
                raise ValueError('Observation changed official FPD IDs or forward count')
        record = output_record(output, prepared, tokenizer)
        if record['unresolved_visible_masks']:
            raise ValueError('Official FPD ended with unresolved visible masks')
        save_trace(directory/'trace.jsonl', trace)
        recorder.finalize()
        if answer:
            answer.finalize(record['visible_tokens'])
        if attention:
            attention.finalize()
        artifacts = ['trace.jsonl', 'drift.jsonl', 'measurement.json']
        if answer:
            artifacts+=['answer_drift.jsonl','answer_measurement.json']
        if attention:
            artifacts+=['attention.jsonl','attention_measurement.json']
        if (directory/'pairwise.npz').exists():
            artifacts.append('pairwise.npz')
        result=dict(id=row['id'], domain=row['domain'], audio_sha256=row['audio_sha256'],
            audio_seconds=row['audio_seconds'], audio_chunks=prepared.audio_chunks, source_index=index, output=record,
            score=grade(record['text'], row['target']), forward_count=nfe,
            untouched_fpd_verified=verified, group_tokens={g:len(p) for g,p in prepared.groups().items()},
            instrumented_seconds=instrumented_seconds, timing_is_instrumented=True,
            artifact_sha256={name:digest(directory/name) for name in artifacts})
        if args.cache_reference_run:
            reference=json.loads((args.cache_reference_run/'samples'/f'{index:05d}'/'baseline.json').read_text())
            verify_cache_reference(result,reference)
            result['clean_cache_baseline_verified']=True
        write_json(result_path,result)
        from .report import analyze
        analyze(args.out)
        completed_this_run += 1
        elapsed = time.monotonic()-started
        print(f'Completed {index+1}/{len(rows)}: {nfe} forwards; '
              f'ETA ~{elapsed/completed_this_run*(len(rows)-index-1)/60:.1f} min '
              '(includes observation/matrix overhead)', flush=True)
    from .report import analyze
    analyze(args.out, plots=True)
    print(f'COMPLETE: {args.out}/analysis/report.md', flush=True)


if __name__ == '__main__':
    main()
