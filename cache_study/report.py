"""Current baseline versus cached autonomous quality, latency, and actual reuse."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from drift_study.storage import write_json


def analyze(out):
    out = Path(out)
    config = json.loads((out/'config.json').read_text())
    names = ['baseline']+list(config['policies'])
    records = {name:{} for name in names}
    for directory in sorted((out/'samples').glob('*')):
        for name in names:
            path = directory/(name+'.json')
            if path.exists():
                record = json.loads(path.read_text())
                records[name][record['id']] = record
    summary, pairs = {}, []
    for name,by_id in records.items():
        rows = list(by_id.values())
        successful = [r for r in rows if not r['failed']]
        seconds = sum(r['seconds'] for r in successful)
        visible = sum(r['output']['visible_tokens'] for r in successful)
        row = dict(n=len(rows),failures=len(rows)-len(successful),
            accuracy=sum(r['score']['correct'] for r in rows)/len(rows) if rows else None,
            parse_failures=sum(r['score']['parse_failed'] for r in rows),
            seconds_successful=seconds,tps=visible/seconds if seconds else None,
            mean_forwards=statistics.mean(r['forward_count'] for r in rows) if rows else None,
            examples_with_actual_reuse=sum(bool(r['accounting'] and r['accounting']['actual_reuse_forwards']) for r in rows))
        paired_seconds,baseline_seconds,losses,recoveries,paired_n = 0.,0.,0,0,0
        for identity,cached in by_id.items():
            if name == 'baseline' or identity not in records['baseline']:
                continue
            baseline = records['baseline'][identity]
            losses += int(baseline['score']['correct'] and not cached['score']['correct'])
            recoveries += int(not baseline['score']['correct'] and cached['score']['correct'])
            valid = not baseline['failed'] and not cached['failed']
            if valid:
                paired_n += 1
                paired_seconds += cached['seconds']
                baseline_seconds += baseline['seconds']
            pairs.append(dict(id=identity,condition=name,domain=cached['domain'],
                baseline_correct=baseline['score']['correct'],cached_correct=cached['score']['correct'],
                cached_failed=cached['failed'],baseline_failed=baseline['failed'],
                baseline_seconds=baseline['seconds'],cached_seconds=cached['seconds'],
                baseline_forwards=baseline['forward_count'],cached_forwards=cached['forward_count'],
                baseline_tokens=baseline['output']['visible_tokens'] if baseline['output'] else 0,
                cached_tokens=cached['output']['visible_tokens'] if cached['output'] else 0,
                actual_reuse_forwards=cached['accounting']['actual_reuse_forwards'],
                skipped_token_update_fraction=cached['accounting']['skipped_token_update_fraction']))
        row.update(paired_successful_n=paired_n,paired_latency_speedup=baseline_seconds/paired_seconds if paired_seconds else None,
                   losses=losses,recoveries=recoveries)
        summary[name] = row
    destination = out/'analysis'
    destination.mkdir(exist_ok=True)
    write_json(destination/'summary.json',dict(conditions=summary,calibration=config['calibration'],
        cache_layers=config['cache_layers'],group_order=config['group_order']))
    if pairs:
        with (destination/'pairs.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    fmt = lambda n:'N/A' if n is None else f'{n:.4f}'
    calibration = config['calibration']
    lines = ['# MMAU-Pro FPD cache benchmark','',
        f'Calibration mean: {calibration["mean_forwards"]}; '
        f'start={calibration["start_mode"]}; dense warmup={calibration["dense_warmup_forwards"]}.',
        f'Cache layers (zero-based block outputs): {config["cache_layers"]}.',
        f'Deep semantic factor: {config.get("deep_semantic_factor",1)}; blocks: {config.get("deep_semantic_layers",[])}.',
        'Schedule order: acoustic / fixed text / semantic. Other layers stay dense.','',
        '| Policy | N | Accuracy | Failures | TPS | Paired latency speedup | Reuse examples | Losses / recoveries |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    if config.get('suite')=='pyramid':
        lines[0]='# MMAU-Pro FPD exploratory depth-pyramid cache benchmark'
        lines[4]='Pyramid suite: explicit per-layer schedules in config.json; no global deep-semantic factor.'
    for name,row in summary.items():
        lines.append(f'| {name} | {row["n"]} | {fmt(row["accuracy"])} | {row["failures"]} | '
            f'{fmt(row["tps"])} | {fmt(row["paired_latency_speedup"])} | '
            f'{row["examples_with_actual_reuse"]} | {row["losses"]} / {row["recoveries"]} |')
    lines += ['','Clean current baseline and all cache policies run on the same GPU with shuffled question-wise condition order.',
        'One synchronized timed autonomous decode per condition/question; audio preparation and scoring are outside timers.',
        'Fixed start is chosen before decoding. Mean-based modes, when used, are development calibration on the same50.',
        'Examples ending during dense warmup receive no cache reuse and remain in the primary accuracy results.',
        'Failed decodes remain incorrect in accuracy; speed/TPS use successful runs, with paired successful counts in summary.json.',
        'Output lengths and forward counts can change; autonomous latency speedup is not equal-work replay.',
        'No drift/attention instrumentation, no per-step activation histories, no sparse attention or pruning.',
        'Warmups use immediate reuse to exercise the cache path, are untimed for reporting, and never enter quality counts.','']
    (destination/'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    analyze(p.parse_args().out)
