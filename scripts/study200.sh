#!/usr/bin/env bash
# 200+200, both refresh profiles matched within each question; one GPU job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset MAX_SECONDS MIN_SECONDS SAMPLES BASELINE_RUN BASELINE_MANIFEST CACHE_REFERENCE_RUN
source scripts/setup.sh
exclusion_args=()
for previous in "${ORIGINAL_MANIFEST:-}" "${LONG_MANIFEST:-}"; do
  if [[ -n "$previous" ]]; then exclusion_args+=(--exclude-manifest "$previous"); fi
done
if [[ ${#exclusion_args[@]} == 0 ]]; then
  echo 'No old manifests supplied: previous-cohort overlap is unchecked (new Vast machine supported).'
fi
short200="${SHORT200_MANIFEST:-$PWD/data/mmau200_short_stereo_v2.json}"
long200="${LONG200_MANIFEST:-$PWD/data/mmau200_long_stereo_v2.json}"
short_output="${SHORT200_RUN:-runs/cache200_short_forward10_paired}"
long_output="${LONG200_RUN:-runs/cache200_long_forward10_paired}"
statistics_output="${STATISTICS_RUN:-runs/mmau200_paired_statistics}"
# Prepare/freeze both cohorts before starting expensive GPU inference.
python -u -m drift_study.data --out "$short200" --samples 200 --seed 20261027 \
  --min-seconds 0 --max-seconds 30 --unique-audio --allow-stereo \
  "${exclusion_args[@]}"
python -u -m drift_study.data --out "$long200" --samples 200 --seed 20261027 \
  --min-seconds 60 --unique-audio --allow-stereo \
  "${exclusion_args[@]}"
python - "$short200" "$long200" "$short_output" "$long_output" "$statistics_output" <<'PY'
import sys
from pathlib import Path
import json
from drift_study.data import read_manifest
from drift_study.storage import digest,write_json
from cache_study.protocol import POLICIES,layer_schedule
short,long=[read_manifest(p) for p in sys.argv[1:3]]
for document in (short,long):
    rows=document['rows']
    if document['settings'].get('audio_channel_policy')!='mono-or-stereo-mean-to-mono-v1':
        raise ValueError('Use the revised mono/stereo manifests for both cohorts')
    if len(rows)!=200 or len({r['audio_sha256'] for r in rows})!=200:
        raise ValueError('Need exactly200 fresh questions with unique audio hashes per cohort')
if any(not 0<r['audio_seconds']<=30 for r in short['rows']):raise ValueError('Short cohort duration mismatch')
if any(r['audio_seconds']<=60 for r in long['rows']) or long['settings']['max_seconds'] is not None:
    raise ValueError('Long cohort must retain complete clips over60s, without a duration cap')
if {r['audio_sha256'] for r in short['rows']}.intersection(r['audio_sha256'] for r in long['rows']):
    raise ValueError('Short and long cohorts overlap in audio')
outputs=[Path(p).resolve() for p in sys.argv[3:]]
if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]):
    raise ValueError('All output folders must be separate and non-nested')
plan=dict(questions_per_cohort=200,unique_audio_per_cohort=200,
    prior_overlap_checked={name:bool(doc['settings'].get('exclude_manifest_sha256')) for name,doc in [('short',short),('long',long)]},
    audio_channel_policy='mono-or-stereo-mean-to-mono-v1',
    manifests={name:digest(path) for name,path in zip(('short','long'),sys.argv[1:3])},
    quotas={'speech':67,'sound':67,'music':66},start_zero_based_forward=10,cache_layers=list(range(32)),
    flat={name:layer_schedule(base,list(range(32))) for name,base in POLICIES.items()},
    halved={name:layer_schedule(base,list(range(32)),list(range(24,32)),2) for name,base in POLICIES.items()},
    conditions_per_question=13,timed_generations=5200,answer_tokens=128,
    primary='12 halved-minus-flat paired comparisons across both cohorts; exact two-sided McNemar,Holm12',
    secondary='24 policy-minus-dense comparisons across both cohorts; Holm24',
    confidence_interval='Conservative95% exact gain/loss difference for unique clips; supplementary20000-draw cluster bootstrap; not multiplicity-adjusted or an exact-test inversion',
    scorer='Unchanged strict final-answer parsing',drift_attention_probes_inside_timers=False)
path=Path('runs/mmau200_experimental_plan.json')
if path.exists():
    if json.loads(path.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('Frozen200 experimental plan changed; use a new checkout')
else:write_json(path,plan)
print('FROZEN:200 unique-audio questions per cohort;13 conditions each;5200 timed generations; see plan for prior-overlap checks',flush=True)
PY
for manifest in "$short200" "$long200"; do
  python -m cache_study.run --manifest "$manifest" --samples 200 --verify-samples 2 \
    --start fixed --start-forward 10 --cache-layers all --deep-semantic-factor 2 \
    --deep-semantic-layers 24-31 --matched-flat --preflight-only
done
echo 'JOB 1/2: short MMAU200; six flat + six halved + clean baseline'
python -u -m cache_study.run --manifest "$short200" --out "$short_output" \
  --diffa-root "$study_diffa" --samples 200 --verify-samples 2 \
  --start fixed --start-forward 10 --cache-layers all --deep-semantic-factor 2 \
  --deep-semantic-layers 24-31 --matched-flat
echo 'JOB 2/2: long MMAU200; same matched conditions'
python -u -m cache_study.run --manifest "$long200" --out "$long_output" \
  --diffa-root "$study_diffa" --samples 200 --verify-samples 2 \
  --start fixed --start-forward 10 --cache-layers all --deep-semantic-factor 2 \
  --deep-semantic-layers 24-31 --matched-flat
python -m cache_study.paired "$short_output" "$long_output" --out "$statistics_output" --expected-samples 200
archive="${RESULTS_ARCHIVE:-runs/mmau200_matched_results.tar.gz}"
python - "$archive" "$short_output" "$long_output" "$statistics_output" <<'PY'
import sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1]);archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as bundle:
    for folder in sys.argv[2:]:
        path=Path(folder);bundle.add(path,arcname=path.name)
    bundle.add('runs/mmau200_experimental_plan.json',arcname='experimental_plan.json')
print(f'ALL200-QUESTION MATCHED RUNS COMPLETE: {archive}',flush=True)
PY
