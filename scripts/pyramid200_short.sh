#!/usr/bin/env bash
# Exploratory depth pyramid on the same frozen short200, with fresh controls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset MAX_SECONDS MIN_SECONDS SAMPLES BASELINE_RUN BASELINE_MANIFEST CACHE_REFERENCE_RUN
source scripts/setup.sh
manifest="${SHORT200_MANIFEST:-$PWD/data/mmau200_short_stereo_v2.json}"
output="${PYRAMID200_RUN:-runs/cache200_short_pyramid_forward10}"
statistics="${PYRAMID_STATISTICS_RUN:-runs/mmau200_short_pyramid_statistics}"
plan="runs/mmau200_short_pyramid_plan.json"
archive="${PYRAMID_RESULTS_ARCHIVE:-runs/mmau200_short_pyramid_results.tar.gz}"
if [[ ! -f "$manifest" ]]; then
  python -u -m drift_study.data --out "$manifest" --samples 200 --seed 20261027 \
    --min-seconds 0 --max-seconds 30 --unique-audio --allow-stereo
fi
python - "$manifest" "$output" "$statistics" "$plan" <<'PY'
import json,sys
from pathlib import Path
from drift_study.data import read_manifest
from drift_study.storage import digest,write_json
from cache_study.protocol import pyramid_specs,PYRAMID_CONTRASTS
m=read_manifest(sys.argv[1]);rows=m['rows']
if len(rows)!=200 or len({r['audio_sha256'] for r in rows})!=200:
    raise ValueError('Need the complete frozen200 unique-audio cohort')
if any(not 0<r['audio_seconds']<=30 for r in rows):
    raise ValueError('Only complete short clips up to30 seconds: one encoder window')
outputs=[Path(p).resolve() for p in sys.argv[2:4]]
if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]):
    raise ValueError('Use separate inference and statistics directories')
specs=pyramid_specs()
plan=dict(suite='pyramid',cohort='short200',exploratory=True,
    note='Motivated by prior results on this cohort; not independent confirmation. No outcome-dependent tuning within this run.',
    questions=200,unique_audio=200,manifest_sha256=digest(sys.argv[1]),audio_selection=m['settings'],
    start_zero_based_forward=10,cache_layers=list(range(32)),group_order=['acoustic','text','semantic'],
    policies=specs,conditions_per_question=7,timed_generations=1400,answer_tokens=128,
    primary_contrasts=PYRAMID_CONTRASTS,primary_correction='Exact two-sided McNemar; Holm across3 exploratory design contrasts',
    secondary='Each of6 cached policies versus its fresh dense baseline; Holm6',
    comparison_note='Flat16 versus equal pyramid compares complete policies; average refresh work is not exactly matched.',
    scorer='Unchanged strict final-answer parsing',drift_attention_probes_inside_timers=False)
path=Path(sys.argv[4])
if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(plan)):
    raise ValueError('Frozen pyramid plan changed; use a separate experiment checkout')
write_json(path,plan)
print('FROZEN:200 short questions,7 conditions,1400 timed generations; exploratory, not a new holdout',flush=True)
for name,spec in specs.items():
    print(name,': A/text/S by groups0–7,8–15,16–23,24–31:',
        [spec['layer_intervals'][layer] for layer in (0,8,16,24)],flush=True)
PY
args=(--manifest "$manifest" --samples 200 --verify-samples 2 --suite pyramid \
  --start fixed --start-forward 10 --cache-layers all)
python -m cache_study.run "${args[@]}" --preflight-only
python -u -m cache_study.run "${args[@]}" --out "$output" --diffa-root "$study_diffa"
python -m cache_study.paired "$output" --out "$statistics" --expected-samples 200 --suite pyramid
python - "$archive" "$output" "$statistics" "$plan" <<'PY'
import sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1]);archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as bundle:
    for folder in sys.argv[2:4]:bundle.add(folder,arcname=Path(folder).name)
    bundle.add(sys.argv[4],arcname='experimental_plan.json')
print(f'PYRAMID STUDY COMPLETE: {archive}',flush=True)
PY
