"""Equal-example attention summaries and per-layer figures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .attention import ATTENTION_GROUPS
from .storage import write_json


def analyze(out,plots=False):
    out=Path(out)
    pooled=defaultdict(list)
    examples,records=0,0
    for result in sorted((out/'samples').glob('*/result.json')):
        path=result.parent/'attention.jsonl'
        if not path.exists():
            continue
        examples+=1
        per_example=defaultdict(list)
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                records+=1
                for metric in ('mass','relative_to_uniform'):
                    for query,keys in row[metric].items():
                        for key in ATTENTION_GROUPS:
                            if metric=='mass' or key in keys:
                                per_example[metric,row['layer'],query,key].append(keys.get(key,0.))
        for cell,values in per_example.items():
            pooled[cell].append(statistics.mean(values))
    if not examples:
        return None
    summary=dict(examples=examples,observed_layer_forward_records=records,
        aggregation='Mean observed forwards per example/layer/query group, then equal example weights',
        warning='Raw modality mass depends on key count; relative_to_uniform accounts for key-group size.',
        mass={},relative_to_uniform={})
    for metric in ('mass','relative_to_uniform'):
        for query in ATTENTION_GROUPS:
            summary[metric][query]={}
            for key in ATTENTION_GROUPS:
                summary[metric][query][key]={str(layer):dict(mean=statistics.mean(values),n=len(values))
                    for (m,layer,q,k),values in pooled.items() if m==metric and q==query and k==key}
    destination=out/'analysis'
    destination.mkdir(exist_ok=True)
    write_json(destination/'attention_summary.json',summary)
    lines=['# Long-audio attention measurements','',
        f'{examples} examples; {records} observed layer/forward records.',
        'FP32 softmax reconstruction from actual post-RoPE Q/K; all heads, keys and queries; self attention included.',
        'Native SDPA results are returned unchanged. Reconstructed probabilities are never used for generation.',
        'Raw mass is the fraction of attention assigned to a key group. Relative-to-uniform divides it by that group’s fraction of sequence positions.',
        'Query groups: acoustic, semantic, fixed text, masked answer, decoded answer. Empty query groups are excluded rather than padded with zero.',
        'Figures show input-key groups; remaining mass goes to masked/decoded answer keys. Full tables retain all five groups.','']
    (destination/'attention_report.md').write_text('\n'.join(lines))
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'acoustic':'#1976d2','semantic':'#e47813','text':'#16805c'}
        for query in ('text','answer_masked','answer_decoded'):
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for ax,metric in zip(axes,('mass','relative_to_uniform')):
                for key in ('acoustic','semantic','text'):
                    cells=summary[metric][query][key]
                    layers=sorted(map(int,cells))
                    ax.plot(layers,[cells[str(l)]['mean'] for l in layers],label=key,color=colors[key],lw=2)
                if metric=='relative_to_uniform':
                    ax.axhline(1,color='#999',ls='--',lw=1)
                ax.set(xlabel='Transformer block (zero-based)',ylabel=metric.replace('_',' '))
                ax.grid(alpha=.2)
                ax.legend()
            fig.suptitle(f'Attention from {query.replace("_"," ")} queries to input modalities')
            fig.savefig(destination/f'attention_{query}.png',dpi=160)
            plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    analyze(args.out,args.plots)
