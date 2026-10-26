"""Equal-example summaries, paired uncertainty, and VidLaDA-style figures."""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import GROUPS, SEED
from .storage import write_json


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return dict(n=len(values), mean=float(values.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist())


def read_sample(directory):
    result = json.loads((directory/'result.json').read_text())
    measurement = json.loads((directory/'measurement.json').read_text())
    records = [json.loads(line) for line in (directory/'drift.jsonl').read_text().splitlines()]
    lookup = {(r['reference'], r['step'], r['layer'], r['group']):r for r in records}
    steps, layers = measurement['steps'], measurement['layers']
    expected = {(ref, t, layer, group) for ref in ('previous', 'initial')
                for t in range(1, steps) for layer in layers for group in GROUPS}
    if not expected <= lookup.keys() or len(lookup) != len(records):
        raise ValueError('Missing or duplicate drift measurements')
    per_layer, all_groups = {}, {}
    for group in GROUPS:
        per_layer[group] = {str(layer):float(np.mean([lookup['previous', t, layer, group]['metrics']['mean']
            for t in range(1, steps)])) for layer in layers} if steps > 1 else {}
        all_groups[group] = float(np.mean(list(per_layer[group].values()))) if steps > 1 else None
    return dict(result=result, measurement=measurement, records=records,
                layer_means=per_layer, group_means=all_groups, directory=directory)


def aggregate(samples):
    usable = [s for s in samples if s['measurement']['steps'] > 1]
    means = {group:float(np.mean([s['group_means'][group] for s in usable])) if usable else None for group in GROUPS}
    paired = {group+'_minus_text':paired_interval([s['group_means'][group]-s['group_means']['text'] for s in usable])
              for group in ('semantic', 'acoustic')}
    layers = sorted(set(layer for s in usable for layer in s['layer_means']['text']), key=int)
    per_layer = {g:{layer:float(np.mean([s['layer_means'][g][layer] for s in usable]))
                    for layer in layers} for g in GROUPS}
    return dict(n=len(samples), n_with_drift=len(usable), no_adjacent_comparison=len(samples)-len(usable),
        accuracy=float(np.mean([s['result']['score']['correct'] for s in samples])) if samples else None,
        parse_failures=sum(s['result']['score']['parse_failed'] for s in samples),
        capacity_exhaustions=sum(s['result']['output']['capacity_exhausted'] for s in samples),
        mean_forwards=float(np.mean([s['result']['forward_count'] for s in samples])) if samples else None,
        group_mean_drift=means, paired_differences=paired, per_layer_mean_drift=per_layer)


def plot(samples, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    usable = [s for s in samples if s['measurement']['steps'] > 1]
    if not usable:
        return
    layers = len(usable[0]['measurement']['layers'])
    max_step = max(s['measurement']['steps'] for s in usable)
    for reference in ('previous', 'initial'):
        raw, normalized, counts = {}, {}, {}
        for group in GROUPS:
            total = np.zeros((layers, max_step))
            count = np.zeros_like(total)
            normalized_examples = []
            for sample in usable:
                steps = sample['measurement']['steps']
                array = np.full((layers, steps), np.nan)
                for row in sample['records']:
                    if row['group'] == group and row['reference'] == reference:
                        array[row['layer'], row['step']] = row['metrics']['mean']
                # Step zero is not a drift comparison, including for initial reference.
                array[:, 0] = np.nan
                valid = np.isfinite(array)
                total[:, :steps] += np.nan_to_num(array)
                count[:, :steps] += valid
                progress = np.arange(1, steps)/(steps-1)
                grid = np.linspace(0, 1, 101)
                interpolated = np.stack([np.interp(grid, progress, a[1:], left=np.nan) for a in array])
                normalized_examples.append(interpolated)
            raw[group] = np.divide(total, count, out=np.full_like(total, np.nan), where=count>0)
            counts[group] = count
            stack = np.stack(normalized_examples)
            n = np.isfinite(stack).sum(axis=0)
            normalized[group] = np.divide(np.nansum(stack, axis=0), n,
                out=np.full(stack.shape[1:], np.nan), where=n>0)
            np.savez_compressed(destination/f'{reference}_{group}_heatmap.npz',
                raw=raw[group], raw_contributor_counts=count, normalized=normalized[group],
                normalized_contributor_counts=n)
        vmax = max(float(np.nanmax(a)) for a in raw.values()) or 1e-6
        for mode, arrays in (('raw', raw), ('normalized', normalized)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for ax, group in zip(axes, GROUPS):
                im = ax.imshow(arrays[group], origin='lower', aspect='auto', vmin=0, vmax=vmax,
                    extent=[0, max_step-1 if mode=='raw' else 1, -.5, layers-.5])
                ax.set(title=group, xlabel='Actual forward index' if mode=='raw' else 'Normalized decoding progress',
                       ylabel='Transformer block (zero-based)')
            fig.colorbar(im, ax=axes, label='Mean per-token cosine distance')
            fig.suptitle(f'{reference} reference — {mode} — equal-example aggregation')
            fig.savefig(destination/f'{reference}_{mode}_drift.png', dpi=160)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
        ax.plot(np.arange(max_step), counts['text'][0])
        ax.set(xlabel='Actual forward index', ylabel='Contributing examples', title='Raw-step coverage')
        fig.savefig(destination/f'{reference}_contributors.png', dpi=140)
        plt.close(fig)
    for sample in samples:
        path = sample['directory']/'pairwise.npz'
        if not path.exists():
            continue
        with np.load(path) as matrices:
            measured = sample['measurement']['pairwise_layers']
            fig, axes = plt.subplots(3, len(measured), figsize=(4*len(measured), 11), squeeze=False,
                                     constrained_layout=True)
            for row, group in enumerate(GROUPS):
                for col, layer in enumerate(measured):
                    ax = axes[row, col]
                    im = ax.imshow(matrices[f'{group}_layer{layer}_token_mean'], origin='lower',
                                   vmin=.8, vmax=1., aspect='equal')
                    ax.set(title=f'{group}, block {layer}', xlabel='Forward index', ylabel='Forward index')
            fig.colorbar(im, ax=axes, label='Mean per-token cosine similarity (display clipped to 0.8–1)')
            fig.suptitle(f'{sample["result"]["domain"]}: {sample["result"]["id"]}')
            fig.savefig(destination/f'pairwise_{sample["result"]["domain"]}.png', dpi=140)
            plt.close(fig)


def analyze(out, plots=False):
    out = Path(out)
    samples = [read_sample(path.parent) for path in sorted((out/'samples').glob('*/result.json'))]
    destination = out/'analysis'
    destination.mkdir(exist_ok=True)
    summary = dict(all=aggregate(samples), domains={domain:aggregate([s for s in samples if s['result']['domain']==domain])
        for domain in ('speech', 'sound', 'music')},
        examples=[dict(id=s['result']['id'], domain=s['result']['domain'], group_mean_drift=s['group_means'],
                       per_layer_mean_drift=s['layer_means']) for s in samples])
    write_json(destination/'summary.json', summary)
    fmt = lambda x:'N/A' if x is None else f'{x:.6f}'
    lines = ['# MMAU-Pro / official FPD drift', '',
        '128 maximum answer slots, 32-token blocks, factor=1.0; variable commits and natural EOS.', '',
        'Long clips use complete chronological encoder windows; acoustic tokens are 64 per window, not 64 per complete clip.', '',
        'Hidden states at all transformer block outputs. Primary: mean per-token cosine distance.',
        'Average steps within each example, then layers, then examples with equal weight.', '',
        '| Cohort | N | N with adjacent drift | Semantic | Acoustic | Fixed text | Accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for domain, row in [('all', summary['all']), *summary['domains'].items()]:
        values = row['group_mean_drift']
        lines.append(f'| {domain} | {row["n"]} | {row["n_with_drift"]} | {fmt(values["semantic"])} | '
                     f'{fmt(values["acoustic"])} | {fmt(values["text"])} | {fmt(row["accuracy"])} |')
    lines += ['', 'Paired differences below: negative means audio drifts less than fixed text.', '']
    for name, value in summary['all']['paired_differences'].items():
        ci = value['ci95']
        lines.append(f'- {name}: {fmt(value["mean"])}; '
                     f'95% paired example-bootstrap interval {ci}; n={value["n"]}.')
    lines += ['', f'Capacity-exhausted answers: {summary["all"]["capacity_exhaustions"]}; '
              f'final-letter parse failures: {summary["all"]["parse_failures"]}.', '',
        'All completed examples, including wrong answers, remain in the primary analysis.',
        'Single-forward examples have no adjacent comparison and are counted separately.',
        'The text group includes fixed chat delimiters; it excludes generated answer positions.',
        'Timings include instrumentation and must not be presented as inference throughput.',
        'VidLaDA does not publish the exact reduction or activation hook for its drift plots. '
        'This study specifies block outputs, token-mean cosine, and flattened cosine as a sensitivity check.',
        'Raw-step figures have contributor-count sidecars; normalized-progress interpolation is a separate view.',
        'Pairwise matrices use predetermined domain exemplars at blocks 0,7,15,23,31; '
        'adjacent and initial-reference measurements cover every block on every example.', '']
    if plots:
        plot(samples, destination)
        from .attention_report import analyze as attention_analyze
        attention_analyze(out,plots=True)
        from .answer_report import build as build_answer_html
        build_answer_html(out)
    (destination/'report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    analyze(args.out, args.plots)
