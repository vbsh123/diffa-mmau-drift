"""Final paired accuracy statistics for matched flat/halved cohort runs."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

from drift_study.storage import write_json,digest
from .protocol import POLICIES,layer_schedule


def exact_mcnemar(gains,losses):
    discordant=gains+losses
    return min(1.,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.


def clip_signflip(deltas):
    """Exact paired label swaps of all question outcomes in each audio cluster."""
    nonzero=[int(v) for v in deltas if v]
    distribution={0:1}
    for delta in nonzero:
        updated=defaultdict(int)
        for total,count in distribution.items():
            updated[total+delta]+=count
            updated[total-delta]+=count
        distribution=updated
    observed=abs(sum(nonzero))
    return sum(count for total,count in distribution.items() if abs(total)>=observed)/2**len(nonzero)


def holm(rows,pkey,target):
    order=sorted(range(len(rows)),key=lambda i:rows[i][pkey]);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(rows)-rank)*rows[index][pkey]))
        rows[index][target]=previous


def comparison(reference,candidate,ids,draws=20000,seed=20261027):
    if set(reference)!=set(ids) or set(candidate)!=set(ids) or len(set(ids))!=len(ids):
        raise ValueError('Paired statistics require exactly the same complete unique question IDs')
    clusters=defaultdict(list);gains=losses=both=neither=0
    for identity in ids:
        a,b=reference[identity],candidate[identity]
        if (a['id'],a['audio_sha256'],a['score']['target'])!=(b['id'],b['audio_sha256'],b['score']['target']):
            raise ValueError('Paired question/audio/reference-answer mismatch')
        x=bool(a['score']['correct']) and not a['failed'];y=bool(b['score']['correct']) and not b['failed']
        gains+=int(y and not x);losses+=int(x and not y);both+=int(x and y);neither+=int(not x and not y)
        clusters[a['audio_sha256']].append(int(y)-int(x))
    delta=np.array([sum(group) for group in clusters.values()],dtype=float)
    sizes=np.array([len(group) for group in clusters.values()],dtype=float)
    rng=np.random.default_rng(seed);boot=[]
    # Bounded temporary arrays for cohorts larger than the original50.
    for start in range(0,draws,1000):
        indices=rng.integers(0,len(clusters),(min(1000,draws-start),len(clusters)))
        boot.extend((100*delta[indices].sum(1)/sizes[indices].sum(1)).tolist())
    discordant=gains+losses
    conditional=binomtest(gains,discordant).proportion_ci(method='exact') if discordant else None
    # Conservative effect bounds remain nonzero-width even with no discordances.
    gain_bounds=binomtest(gains,len(ids)).proportion_ci(confidence_level=.975,method='exact')
    loss_bounds=binomtest(losses,len(ids)).proportion_ci(confidence_level=.975,method='exact')
    conservative=[100*(gain_bounds.low-loss_bounds.high),100*(gain_bounds.high-loss_bounds.low)]
    bootstrap=np.quantile(boot,[.025,.975]).tolist()
    effect_ci=conservative if len(clusters)==len(ids) else bootstrap
    paired_success=[(reference[i],candidate[i]) for i in ids
        if not reference[i]['failed'] and not candidate[i]['failed']]
    def tps(rows):
        successful=[r for r in rows.values() if not r['failed']]
        seconds=sum(r['seconds'] for r in successful)
        return sum(r['output']['visible_tokens'] for r in successful)/seconds if seconds else None
    rtps,ctps=tps(reference),tps(candidate)
    paired_seconds=sum(c['seconds'] for a,c in paired_success)
    speedup=sum(a['seconds'] for a,c in paired_success)/paired_seconds if paired_seconds else None
    p=exact_mcnemar(gains,losses)
    if abs(p-(binomtest(gains,discordant,.5).pvalue if discordant else 1.))>1e-12:
        raise ValueError('Exact McNemar validation failed')
    return dict(n=len(ids),audio_clusters=len(clusters),reference_tps=rtps,candidate_tps=ctps,
        tps_gain_vs_reference_percent=100*(ctps/rtps-1) if rtps and ctps else None,
        paired_latency_speedup=speedup,paired_timing_n=len(paired_success),baseline_correct=both+losses,candidate_correct=both+gains,
        gains=gains,losses=losses,both_correct=both,both_wrong=neither,
        accuracy_delta_pp=100*(gains-losses)/len(ids),mcnemar_exact_p=p,clip_p=clip_signflip(delta),
        accuracy_delta_cluster_bootstrap95_pp=bootstrap,accuracy_delta_conservative95_pp=conservative,
        accuracy_delta_ci95_pp=effect_ci,accuracy_delta_ci_method='Conservative exact gain/loss bounds' if len(clusters)==len(ids) else 'Approximate audio-cluster bootstrap',
        conditional_gain_share_exact95=[conditional.low,conditional.high] if conditional else None,
        failed_reference=sum(r['failed'] for r in reference.values()),failed_candidate=sum(r['failed'] for r in candidate.values()))


def analyze(runs,out,expected_samples=200,draws=20000):
    all_rows=[];cohorts=[];signature=None
    for run in map(Path,runs):
        config=json.loads((run/'config.json').read_text());ids=config['sample_ids']
        if len(ids)!=expected_samples:
            raise ValueError(f'Need all{expected_samples} questions, got{len(ids)} in {run}')
        frozen=run/'cohort.json'
        if digest(frozen)!=config['manifest_sha256']:
            raise ValueError('Frozen cohort manifest hash mismatch')
        cohort=json.loads(frozen.read_text());truth={r['id']:r for r in cohort['rows']}
        if [r['id'] for r in cohort['rows']]!=ids:
            raise ValueError('Frozen cohort IDs/order differ from run configuration')
        policies=config['policies'];conditions=['baseline']+list(policies)
        expected={name:base for name,base in POLICIES.items()}
        expected.update({name+'_deepsem2':base for name,base in POLICIES.items()})
        if set(policies)!=set(expected) or any(tuple(policies[name])!=base for name,base in expected.items()):
            raise ValueError('Need all six flat and six matched halved policies')
        if (config['cache_layers']!=list(range(32)) or config['deep_semantic_factor']!=2
                or config['deep_semantic_layers']!=list(range(24,32))
                or config['calibration']['start_mode']!='fixed'
                or config['calibration']['dense_warmup_forwards']!=10):
            raise ValueError('Matched study requires all32 layers,forward10,and semantic factor2 in24–31')
        for name,base in expected.items():
            schedule=layer_schedule(base,list(range(32)),list(range(24,32)) if name.endswith('_deepsem2') else (),2 if name.endswith('_deepsem2') else 1)
            if config['layer_intervals'][name]!=json.loads(json.dumps(schedule)):
                raise ValueError('Saved per-layer schedule differs from the matched study')
        if any(config['decoder'].get(key)!=value for key,value in
               dict(max_answer_tokens=128,block_tokens=32,factor=1.,temperature=0.).items()):
            raise ValueError('Study decoder settings differ from official FPD128 configuration')
        current={key:config[key] for key in ('models','decoder','seed','source_sha256')}
        if signature is not None and current!=signature:
            raise ValueError('Cohorts use different model/decoder/source/seed settings')
        signature=current
        by={condition:{} for condition in conditions}
        for directory in sorted((run/'samples').glob('*')):
            for condition in conditions:
                path=directory/(condition+'.json')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['id'] in by[condition]:raise ValueError('Duplicate question record')
                    expected_row=truth.get(record['id'])
                    if expected_row is None or (record['audio_sha256'],record['score']['target'])!=(expected_row['audio_sha256'],expected_row['target']):
                        raise ValueError('Saved question/audio/target differs from frozen cohort')
                    by[condition][record['id']]=record
        if any(set(records)!=set(ids) for records in by.values()):
            raise ValueError('Run incomplete: need every condition on the same full cohort')
        cohorts.append(dict(run=str(run),n=len(ids),conditions=len(conditions),
            manifest_sha256=config['manifest_sha256'],settings=config['calibration']))
        for name in policies:
            row=comparison(by['baseline'],by[name],ids,draws=draws)
            all_rows.append(dict(cohort=run.name,family='versus_dense',reference='baseline',candidate=name,**row))
        flat=[name for name in policies if '_deepsem' not in name]
        for name in flat:
            counterpart=name+'_deepsem2'
            if counterpart not in policies:raise ValueError(f'Missing matched halved policy for{name}')
            row=comparison(by[name],by[counterpart],ids,draws=draws)
            all_rows.append(dict(cohort=run.name,family='halved_vs_flat',reference=name,candidate=counterpart,**row))
    for family in ('halved_vs_flat','versus_dense'):
        selected=[r for r in all_rows if r['family']==family]
        holm(selected,'mcnemar_exact_p','holm_family_p')
        holm(selected,'clip_p','holm_family_clip_p')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=dict(cohorts=cohorts,comparisons=all_rows,bootstrap_draws=draws,environment=dict(numpy=np.__version__,scipy=scipy.__version__),
        primary='Matched halved versus flat; Holm across12 tests when both cohorts are supplied',
        secondary='Each policy versus dense baseline; Holm across24 tests when both cohorts are supplied',
        interval='Unique clips: conservative95% difference of97.5% exact binomial gain/loss bounds. Bootstrap supplementary; cluster bootstrap if clips repeat. Not multiplicity-adjusted or an exact-test inversion.',
        null='Two-sided equality; no equivalence/noninferiority claim or post-hoc one-sided test',
        scorer='Original strict final-answer scores; no rescoring',expected_samples=expected_samples)
    write_json(out/'paired_statistics.json',report)
    with (out/'paired_statistics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched caching paired statistics</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:20px;background:#f5f7fb;color:#203047}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}.table{overflow:auto}select{font:inherit;padding:8px}p{line-height:1.5}</style></head><body><h1>Matched flat/halved cache results</h1><p>All conditions use the same frozen questions in each cohort. Gains/losses compare candidate against reference. Exact two-sided McNemar tests are primary for unique clips; clip-level paired label swaps preserve repeated audio clusters. Accuracy intervals use conservative exact gain/loss bounds for unique clips, with a supplementary cluster bootstrap. If audio repeats, displayed intervals use the approximate cluster bootstrap. No interval is multiplicity-adjusted or an exact-test inversion. Failure to reject equality does not demonstrate equivalence or safety.</p><label>Family <select id="family"><option value="halved_vs_flat">Primary: halved versus flat</option><option value="versus_dense">Secondary: versus dense baseline</option></select></label><div class="table"><table><thead><tr><th>Cohort</th><th>Reference</th><th>Candidate</th><th>N / clips</th><th>Accuracy reference→candidate</th><th>Δpp [95% CI]</th><th>Gains / losses</th><th>Exact p</th><th>Holm p</th><th>Clip p</th><th>Holm clip p</th><th>TPS reference→candidate</th><th>TPS Δ vs reference</th><th>Paired latency speedup</th></tr></thead><tbody id="rows"></tbody></table></div><script>const DATA=__DATA__;const fmt=(v,d=2)=>v==null?'N/A':v.toFixed(d);function update(){document.getElementById('rows').innerHTML=DATA.comparisons.filter(r=>r.family===document.getElementById('family').value).map(r=>`<tr><td>${r.cohort}</td><td>${r.reference}</td><td>${r.candidate}</td><td>${r.n} / ${r.audio_clusters}</td><td>${(100*r.baseline_correct/r.n).toFixed(1)}%→${(100*r.candidate_correct/r.n).toFixed(1)}%</td><td>${r.accuracy_delta_pp.toFixed(1)} [${r.accuracy_delta_ci95_pp.map(v=>v.toFixed(1)).join(', ')}]</td><td>${r.gains} / ${r.losses}</td><td>${r.mcnemar_exact_p.toPrecision(3)}</td><td>${r.holm_family_p.toPrecision(3)}</td><td>${r.clip_p.toPrecision(3)}</td><td>${r.holm_family_clip_p.toPrecision(3)}</td><td>${fmt(r.reference_tps)}→${fmt(r.candidate_tps)}</td><td>${fmt(r.tps_gain_vs_reference_percent,1)}%</td><td>${fmt(r.paired_latency_speedup,3)}×</td></tr>`).join('')}document.getElementById('family').onchange=update;update();window.PAIRED_RESULTS=DATA;</script></body></html>'''
    (out/'paired_statistics.html').write_text(html.replace('__DATA__',json.dumps(report,allow_nan=False).replace('<','\\u003c')))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--expected-samples',type=int,default=200)
    parser.add_argument('--draws',type=int,default=20000);args=parser.parse_args()
    analyze(args.runs,args.out,args.expected_samples,args.draws)
    print(f'PAIRED STATISTICS COMPLETE: {args.out}/paired_statistics.html')
