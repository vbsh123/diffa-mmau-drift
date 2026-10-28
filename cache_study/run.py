"""Six cache schedules, same 50 questions, clean autonomous FPD timing."""
import argparse
from contextlib import ExitStack, contextmanager
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import torch

from drift_study.checkout import COMMIT, DEFAULT_ROOT, verify_checkout
from drift_study.audio_windows import INPUT_PROTOCOL
from drift_study.data import load_audio, read_manifest
from drift_study.fpd import generate, output_record
from drift_study.model import DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION, load_model, prepare
from drift_study.protocol import ANSWER_TOKENS, BLOCK_TOKENS, FACTOR, SEED, grade
from drift_study.run import WHISPER, WHISPER_REVISION
from drift_study.storage import digest, write_json
from .engine import PrefixCache
from .protocol import (GROUP_ORDER,POLICIES,benchmark_source_digest,calibrate,parse_layers,
                       layer_schedule,fixed_start_settings)


class ForwardLimit(RuntimeError):
    pass


@contextmanager
def watchdog(llm, limit=1024):
    counter = dict(forwards=0)
    def before(module, args):
        if counter['forwards'] >= limit:
            raise ForwardLimit(f'Official FPD exceeded {limit} forwards; no forced commits')
        counter['forwards'] += 1
    handle = llm.register_forward_pre_hook(before)
    try:
        yield counter
    finally:
        handle.remove()


def measure(model,prepared,tokenizer,intervals,layers,start_forward,layer_intervals=None):
    engine=PrefixCache(model.llm_model,prepared,intervals,layers,start_forward,layer_intervals) if intervals else None
    output, failure = None, None
    torch.cuda.synchronize()
    began = time.perf_counter()
    try:
        with ExitStack() as stack:
            counter = stack.enter_context(watchdog(model.llm_model))
            if engine:
                stack.enter_context(engine)
            output, nfe, _ = generate(model,prepared,trace=False)
    except ForwardLimit as error:
        failure = str(error)
        nfe = counter['forwards']
    torch.cuda.synchronize()
    seconds = time.perf_counter()-began
    record = output_record(output,prepared,tokenizer) if output is not None else None
    if record is not None and record['unresolved_visible_masks']:
        failure = 'Official FPD ended with unresolved visible masks'
    return dict(seconds=seconds,forward_count=nfe,output=record,failed=failure is not None,
                failure=failure,accounting=engine.accounting() if engine else None)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path)
    p.add_argument('--baseline-manifest',type=Path)
    p.add_argument('--manifest',type=Path,default=Path('data/mmau50_long.json'))
    p.add_argument('--out',type=Path,default=Path('runs/cache50_long_forward10_all_deepsem2_24-31'))
    p.add_argument('--diffa-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--policies',nargs='+',choices=list(POLICIES),default=list(POLICIES))
    p.add_argument('--cache-layers',default='all',help='all, 16-31, or 16,20,24; zero-based blocks')
    p.add_argument('--start',choices=['fixed','mean','half-mean','immediate'],default='fixed')
    p.add_argument('--start-forward',type=int,default=10)
    p.add_argument('--deep-semantic-factor',type=int,default=2)
    p.add_argument('--deep-semantic-layers',default='24-31')
    p.add_argument('--allow-legacy-short-cohort',action='store_true',help='Explicitly reuse an existing frozen short-audio manifest; does not select new short clips')
    p.add_argument('--matched-flat',action='store_true',help='Also run the six unmodified flat controls')
    p.add_argument('--samples',type=int,default=50,help='First N of the frozen cohort; matched study uses200 per arm')
    p.add_argument('--verify-samples',type=int,default=2)
    p.add_argument('--preflight-only',action='store_true',help='Validate full calibration without loading models')
    return p


def main():
    args = parser().parse_args()
    if args.samples<1 or not 0 <= args.verify_samples <= args.samples:
        raise ValueError('Invalid samples/verification count')
    if len(set(args.policies)) != len(args.policies):
        raise ValueError('Duplicate cache policy')
    manifest = read_manifest(args.manifest)
    if args.samples>len(manifest['rows']) or (manifest['settings'].get('input_protocol')!=INPUT_PROTOCOL and not args.allow_legacy_short_cohort):
        raise ValueError('Requested samples exceed the frozen cohort, or input protocol is unsupported')
    if args.baseline_run:
        old_manifest=args.baseline_manifest or args.baseline_run.resolve().parents[1]/'data'/'mmau50_long.json'
        calibration=calibrate(args.baseline_run,old_manifest,manifest,args.start,args.start_forward)
    elif args.start=='fixed':
        calibration=fixed_start_settings(args.start_forward)
    else:
        raise ValueError('Mean-based calibration requires --baseline-run; fixed forward10 does not')
    layers = parse_layers(args.cache_layers)
    deep_layers=parse_layers(args.deep_semantic_layers)
    specs={}
    for name in args.policies:
        base=POLICIES[name]
        if args.matched_flat or args.deep_semantic_factor==1:
            specs[name]=dict(base=base,layer_intervals=layer_schedule(base,layers))
        if args.deep_semantic_factor!=1:
            specs[f'{name}_deepsem{args.deep_semantic_factor}']=dict(base=base,
                layer_intervals=layer_schedule(base,layers,deep_layers,args.deep_semantic_factor))
    expected_models = dict(diffa=[DIFFA_REPO,DIFFA_REVISION],llm=[LLM_REPO,LLM_REVISION],
        whisper=[WHISPER,WHISPER_REVISION],lora_alpha=16)
    if 'source_models' in calibration and calibration['source_models'] != expected_models:
        raise ValueError('Original model revisions differ from cache benchmark')
    print(f'Frozen start: mean={calibration["mean_forwards"]}; '
          f'dense warmup={calibration["dense_warmup_forwards"]}; layers={layers}',flush=True)
    if args.preflight_only:
        print('PREFLIGHT PASSED; no model loaded',flush=True)
        return
    verify_checkout(args.diffa_root)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run on a BF16 CUDA GPU')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = manifest['rows'][:args.samples]
    config = dict(protocol='mmau-fpd128-layerwise-cache-v2',upstream_commit=COMMIT,
        source_sha256=benchmark_source_digest(),manifest_sha256=digest(args.manifest),
        sample_ids=[r['id'] for r in rows],models=expected_models,
        input_protocol=manifest['settings'].get('input_protocol','legacy-short'),
        decoder=dict(max_answer_tokens=ANSWER_TOKENS,block_tokens=BLOCK_TOKENS,factor=FACTOR,temperature=0.),
        policies={name:spec['base'] for name,spec in specs.items()},group_order=GROUP_ORDER,
        layer_intervals={name:spec['layer_intervals'] for name,spec in specs.items()},
        deep_semantic_factor=args.deep_semantic_factor,deep_semantic_layers=deep_layers,
        cache_layers=layers,calibration=calibration,seed=SEED,verify_samples=args.verify_samples,
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            transformers=importlib.metadata.version('transformers'),gpu=torch.cuda.get_device_name(0)),
        timing='One clean synchronized autonomous generation per condition/question; shuffled order; no drift hooks.')
    # Roundtrip tuples so resume comparison matches JSON exactly.
    config = json.loads(json.dumps(config))
    args.out.mkdir(parents=True,exist_ok=True)
    config_path = args.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Cache config/source/environment/calibration changed; use a new output directory')
    elif any(args.out.iterdir()):
        raise ValueError('Output directory lacks a matching cache config')
    else:
        write_json(config_path,config)
    write_json(args.out/'calibration.json',calibration)
    frozen_manifest=args.out/'cohort.json'
    if frozen_manifest.exists():
        if digest(frozen_manifest)!=config['manifest_sha256']:
            raise ValueError('Frozen output cohort manifest was changed')
    else:
        shutil.copyfile(args.manifest,frozen_manifest)
    model,tokenizer,extractor,load_report = load_model(args.diffa_root,
        DIFFA_REPO,DIFFA_REVISION,LLM_REPO,LLM_REVISION,WHISPER,WHISPER_REVISION)
    write_json(args.out/'model_load.json',load_report)
    if len(model.llm_model.model.transformer.blocks) != 32 or tokenizer.eos_token_id != 126081:
        raise ValueError('Unexpected checkpoint architecture or EOS')
    start_forward = calibration['dense_warmup_forwards']
    warmed,finished,started = False,0,time.monotonic()
    conditions=['baseline']+list(specs)
    for index,row in enumerate(rows):
        directory = args.out/'samples'/f'{index:05d}'
        directory.mkdir(parents=True,exist_ok=True)
        order = conditions.copy()
        random.Random(SEED+index).shuffle(order)
        missing = [name for name in order if not (directory/(name+'.json')).exists()]
        saved = {}
        for name in order:
            path = directory/(name+'.json')
            if path.exists():
                item = json.loads(path.read_text())
                if item['id'] != row['id'] or item['audio_sha256'] != row['audio_sha256']:
                    raise ValueError('Existing cache sample identity mismatch')
                saved[name] = item
        needs_verification = index < args.verify_samples and not (directory/'verification.json').exists()
        if not missing and not needs_verification:
            print(f'Skip complete {index+1}/{len(rows)}',flush=True)
            continue
        audio = load_audio(args.manifest.resolve().parent/row['audio'],
            allow_stereo=manifest['settings'].get('audio_channel_policy')=='mono-or-stereo-mean-to-mono-v1')
        prepared = prepare(model,tokenizer,extractor,audio,row['prompt'],ANSWER_TOKENS)
        if not warmed:
            print('Process warmup: dense and each cache policy once; cache warmups use immediate reuse',flush=True)
            for name in conditions:
                spec=specs.get(name)
                warm=measure(model,prepared,tokenizer,spec['base'] if spec else None,layers,0,
                             spec['layer_intervals'] if spec else None)
                if warm['failed']:
                    raise RuntimeError(f'Warmup failed for {name}: {warm["failure"]}')
            warmed = True
        for name in order:
            if name in saved:
                continue
            print(f'Question {index+1}/{len(rows)}: {name}, start={start_forward}, layers={args.cache_layers}',flush=True)
            spec=specs.get(name)
            measured=measure(model,prepared,tokenizer,spec['base'] if spec else None,layers,start_forward,
                             spec['layer_intervals'] if spec else None)
            text = measured['output']['text'] if measured['output'] else ''
            measured.update(condition=name,id=row['id'],domain=row['domain'],
                audio_sha256=row['audio_sha256'],audio_seconds=row['audio_seconds'],audio_chunks=prepared.audio_chunks,
                score=grade(text,row['target']),group_tokens={g:len(p) for g,p in prepared.groups().items()},
                condition_order=order)
            if measured['failed']:
                measured['score']['correct'] = False
            if name=='baseline' and args.baseline_run:
                old = json.loads((args.baseline_run/'samples'/f'{index:05d}'/'result.json').read_text())
                measured['matches_original_drift_canvas'] = bool(measured['output'] and
                    measured['output']['canvas_ids'] == old['output']['canvas_ids'])
            saved[name] = measured
            write_json(directory/(name+'.json'),measured)
            from .report import analyze
            analyze(args.out)
        if needs_verification:
            identity = measure(model,prepared,tokenizer,(1,1,1),layers,0)
            baseline = saved['baseline']
            match = bool(not identity['failed'] and not baseline['failed'] and
                identity['forward_count'] == baseline['forward_count'] and
                identity['output']['canvas_ids'] == baseline['output']['canvas_ids'])
            if not match:
                raise ValueError('Refresh-every-forward cache does not reproduce clean FPD baseline')
            write_json(directory/'verification.json',dict(identity_exact_ids_and_nfe=True,
                cache_layers=layers,identity_forward_count=identity['forward_count']))
        finished += 1
        print(f'Completed {index+1}/{len(rows)}; ETA ~{(time.monotonic()-started)/finished*(len(rows)-index-1)/60:.1f} min',flush=True)
    from .report import analyze
    analyze(args.out)
    print(f'CACHE COMPLETE: {args.out}/analysis/report.md',flush=True)


if __name__ == '__main__':
    main()
