"""Cache schedules and calibration from the complete uncached drift cohort."""
import hashlib
import json
import math
from pathlib import Path

from drift_study.data import read_manifest
from drift_study.protocol import ANSWER_TOKENS, BLOCK_TOKENS, FACTOR, SEED
from drift_study.storage import digest, source_digest

POLICIES = {
    '161616': (16,16,16),
    '888': (8,8,8),
    '16816': (16,8,16),
    '16168': (16,16,8),
    '323216': (32,32,16),
    '321632': (32,16,32),
}
GROUP_ORDER = ('acoustic', 'text', 'semantic')


def pyramid_specs():
    """Fixed small exploratory suite; intervals are A/text/S at each block."""
    layers=list(range(32))
    specs={name:dict(base=POLICIES[name],layer_intervals=layer_schedule(POLICIES[name],layers))
           for name in ('161616','16816')}
    specs['888_deepsem2']=dict(base=POLICIES['888'],
        layer_intervals=layer_schedule(POLICIES['888'],layers,range(24,32),2))
    for name,numerator,denominator in [('pyramid_equal',1,1),('pyramid_audio_slow',2,1),
                                        ('pyramid_audio_fast',1,2)]:
        schedule={layer:(value*numerator//denominator,value,value*numerator//denominator)
                  for layer in layers for value in [(64,32,16,8)[layer//8]]}
        specs[name]=dict(base=schedule[0],layer_intervals=schedule)
    return specs


PYRAMID_CONTRASTS=(('161616','pyramid_equal'),
                   ('pyramid_equal','pyramid_audio_slow'),
                   ('pyramid_equal','pyramid_audio_fast'))


def parse_layers(value, count=32):
    if value == 'all':
        return list(range(count))
    selected = set()
    try:
        for item in value.split(','):
            if '-' in item:
                lo, hi = map(int, item.split('-'))
                if lo > hi:
                    raise ValueError('Reversed layer range')
                selected.update(range(lo, hi+1))
            else:
                selected.add(int(item))
    except (ValueError, TypeError):
        raise ValueError('Use all, a range such as 16-31, or a list such as 16,20,24') from None
    if not selected or min(selected)<0 or max(selected)>=count:
        raise ValueError(f'Cache layers must be within 0–{count-1}')
    return sorted(selected)


def start_from_mean(mean,mode,fixed_forward=10):
    if mode=='fixed':
        if fixed_forward<0:
            raise ValueError('Fixed start forward must be nonnegative')
        return fixed_forward
    if mode == 'immediate':
        return 0
    if mode == 'mean':
        return math.ceil(mean)
    if mode == 'half-mean':
        return math.ceil(mean/2)
    raise ValueError('Unknown cache start mode')


def benchmark_source_digest():
    hasher = hashlib.sha256(source_digest().encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def calibrate(baseline_run,baseline_manifest,manifest,mode='mean',fixed_forward=10):
    """Require the complete original 50; do not estimate from partial results."""
    baseline_run, baseline_manifest = Path(baseline_run), Path(baseline_manifest)
    config = json.loads((baseline_run/'config.json').read_text())
    old_manifest = read_manifest(baseline_manifest)
    rows = manifest['rows']
    if len(rows) != 50 or len(old_manifest['rows']) != 50:
        raise ValueError('Calibration requires the exact complete 50-question cohort')
    if digest(baseline_manifest) != config['manifest_sha256']:
        raise ValueError('Original baseline manifest hash does not match drift config')
    if (manifest['dataset'],manifest['revision']) != (config['dataset'],config['dataset_revision']):
        raise ValueError('Dataset revision mismatch')
    fields = ('id','audio_sha256','prompt','target','domain','audio_seconds')
    if [[r[k] for k in fields] for r in rows] != [[r[k] for k in fields] for r in old_manifest['rows']]:
        raise ValueError('Cache cohort differs from the original drift cohort')
    if config['sample_ids'] != [r['id'] for r in rows]:
        raise ValueError('Original drift run did not select exactly these 50 IDs')
    decoder = config['decoder']
    if (decoder['max_answer_tokens'],decoder['block_tokens'],decoder['factor'],decoder['temperature']) != (ANSWER_TOKENS,BLOCK_TOKENS,FACTOR,0.):
        raise ValueError('Original FPD settings differ')
    if config['source_sha256'] != source_digest() or config['seed'] != SEED:
        raise ValueError('Original drift source or seed differs; do not mix decoder implementations')
    counts, records = [], []
    for index,row in enumerate(rows):
        result_path = baseline_run/'samples'/f'{index:05d}'/'result.json'
        if not result_path.exists():
            raise ValueError(f'Need all 50 completed drift examples before mean calibration; missing {index+1}/50')
        saved = json.loads(result_path.read_text())
        if saved['id'] != row['id'] or saved['audio_sha256'] != row['audio_sha256']:
            raise ValueError('Original drift sample identity mismatch')
        nfe = saved['forward_count']
        if not isinstance(nfe,int) or isinstance(nfe,bool) or not 1 <= nfe <= 1024:
            raise ValueError('Invalid original forward count')
        trace = result_path.parent/'trace.jsonl'
        if digest(trace) != saved['artifact_sha256']['trace.jsonl']:
            raise ValueError('Original forward trace hash mismatch')
        if len(trace.read_text().splitlines()) != nfe:
            raise ValueError('Original trace does not match forward count')
        counts.append(nfe)
        records.append(dict(id=row['id'],forward_count=nfe,result_sha256=digest(result_path)))
    mean = sum(counts)/len(counts)
    return dict(n=50, mean_forwards=mean, start_mode=mode,
        dense_warmup_forwards=start_from_mean(mean,mode,fixed_forward),
        first_reuse_zero_based_forward=max(1,start_from_mean(mean,mode,fixed_forward)),
        calibration_config_sha256=digest(baseline_run/'config.json'),
        source_models=config['models'], records=records,
        note='Development calibration uses the same 50 questions; not held-out or per-example EOS-aware scheduling.')


def layer_schedule(base,layers,deep_layers=(),semantic_factor=1):
    if semantic_factor<1 or base[2]%semantic_factor:
        raise ValueError('Semantic factor must divide the base semantic interval exactly')
    return {layer:(base[0],base[1],base[2]//semantic_factor if layer in deep_layers else base[2])
            for layer in layers}


def fixed_start_settings(forward=10):
    value=start_from_mean(None,'fixed',forward)
    return dict(n=0,mean_forwards=None,start_mode='fixed',dense_warmup_forwards=value,
        first_reuse_zero_based_forward=max(1,value),records=[],
        note='Fixed start chosen before decoding; no mean or per-example future EOS is used.')
