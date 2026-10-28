#!/usr/bin/env bash
# Single-window MMAU200: dense + six flat + six matched deep-semantic halvings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset MAX_SECONDS MIN_SECONDS SAMPLES BASELINE_RUN BASELINE_MANIFEST CACHE_REFERENCE_RUN
source scripts/setup.sh
manifest="${SHORT200_MANIFEST:-$PWD/data/mmau200_short_stereo_v2.json}"
output="${SHORT200_RUN:-runs/cache200_short_only_forward10_paired}"
statistics="${STATISTICS_RUN:-runs/mmau200_short_only_statistics}"
plan="runs/mmau200_short_only_plan.json"
archive="${RESULTS_ARCHIVE:-runs/mmau200_short_only_results.tar.gz}"
# No dependency on previous-machine manifests, and no long cohort preparation.
python -u -m drift_study.data --out "$manifest" --samples 200 --seed 20261027 \
  --min-seconds 0 --max-seconds 30 --unique-audio --allow-stereo
python - "$manifest" "$output" "$statistics" "$plan" <<'PY'
import json,sys
from pathlib import Path
from drift_study.data import read_manifest
from drift_study.storage import digest,write_json
from cache_study.protocol import POLICIES,layer_schedule
m=read_manifest(sys.argv[1]);rows=m['rows']
if len(rows)!=200 or len({r['audio_sha256'] for r in rows})!=200:
    raise ValueError('Need200 questions with unique audio hashes')
if any(not 0<r['audio_seconds']<=30 for r in rows):
    raise ValueError('Short-only study requires complete clips at most30 seconds: one encoder window')
if m['settings'].get('exclude_manifest_sha256'):
    raise ValueError('This protocol uses no previous-cohort exclusions; choose the default manifest')
outputs=[Path(p).resolve() for p in sys.argv[2:4]]
if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]):
    raise ValueError('Inference and statistics directories must be separate and non-nested')
plan=dict(cohort='short_only',questions=200,unique_audio=200,prior_overlap_checked=False,
    manifest_sha256=digest(sys.argv[1]),audio_selection=m['settings'],
    start_zero_based_forward=10,cache_layers=list(range(32)),
    flat={name:layer_schedule(base,list(range(32))) for name,base in POLICIES.items()},
    halved={name:layer_schedule(base,list(range(32)),list(range(24,32)),2) for name,base in POLICIES.items()},
    conditions_per_question=13,timed_generations=2600,answer_tokens=128,
    primary='6 halved-minus-flat paired comparisons; exact two-sided McNemar,Holm6',
    secondary='12 policy-minus-dense comparisons; Holm12',
    scorer='Unchanged strict final-answer parsing',drift_attention_probes_inside_timers=False)
path=Path(sys.argv[4])
if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(plan)):
    raise ValueError('Frozen short-only experimental plan changed')
write_json(path,plan)
print('FROZEN: short-only200; one audio window per question;2600 timed generations; previous overlap unchecked',flush=True)
PY
args=(--manifest "$manifest" --samples 200 --verify-samples 2 --start fixed \
  --start-forward 10 --cache-layers all --deep-semantic-factor 2 \
  --deep-semantic-layers 24-31 --matched-flat)
python -m cache_study.run "${args[@]}" --preflight-only
python -u -m cache_study.run "${args[@]}" --out "$output" --diffa-root "$study_diffa"
python -m cache_study.paired "$output" --out "$statistics" --expected-samples 200
python - "$archive" "$output" "$statistics" "$plan" <<'PY'
import sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1]);archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as bundle:
    for folder in sys.argv[2:4]:bundle.add(folder,arcname=Path(folder).name)
    bundle.add(sys.argv[4],arcname='experimental_plan.json')
print(f'SHORT-ONLY STUDY COMPLETE: {archive}',flush=True)
PY
