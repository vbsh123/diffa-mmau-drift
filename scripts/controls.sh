#!/usr/bin/env bash
# Missing forward-10 controls only: six flat schedules on each frozen cohort.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset MAX_SECONDS BASELINE_RUN BASELINE_MANIFEST CACHE_REFERENCE_RUN
export SAMPLES=50 MIN_SECONDS=60 CACHE_LAYERS=all VERIFY_SAMPLES=2
export CACHE_START=fixed CACHE_START_FORWARD=10 DEEP_SEMANTIC_FACTOR=1 DEEP_SEMANTIC_LAYERS=24-31 MATCHED_FLAT=0
source scripts/setup.sh
export SKIP_INSTALL=1 AUDIO_VENV="$study_venv" DIFFA_ROOT="$study_diffa"
original_manifest="${ORIGINAL_MANIFEST:-/workspace/diffa-mmau-drift/data/mmau50.json}"
long_manifest="${LONG_MANIFEST:-/workspace/diffa-mmau-next/data/mmau50_long.json}"
original_output="${ORIGINAL_CACHE_RUN_NAME:-runs/cache50_original_forward10_all_flat}"
long_output="${LONG_CACHE_RUN_NAME:-runs/cache50_long_forward10_all_flat}"
python - "$original_manifest" "$long_manifest" "$original_output" "$long_output" <<'PY'
import sys
from pathlib import Path
from drift_study.data import read_manifest
from drift_study.audio_windows import INPUT_PROTOCOL
paths=[Path(p).resolve() for p in sys.argv[1:]]
if paths[0]==paths[1] or paths[2]==paths[3] or paths[2] in paths[3].parents or paths[3] in paths[2].parents:
    raise ValueError('Cohorts and result folders must be separate')
original,long=[read_manifest(p) for p in paths[:2]]
if any(len(m['rows'])!=50 for m in (original,long)):
    raise ValueError('Both original frozen 50-question manifests are required')
settings=long['settings']
if (settings.get('input_protocol')!=INPUT_PROTOCOL or settings.get('min_seconds')!=60
        or settings.get('max_seconds') is not None):
    raise ValueError('Use the same existing uncapped long cohort; no resampling')
print('Both frozen cohorts verified: 50 questions each, all WAV hashes/prompts intact',flush=True)
PY
echo 'CONTROL JOB 1/2: original MMAU50, forward10, six unhalved schedules'
env MANIFEST="$original_manifest" RUN_NAME="$original_output" ALLOW_LEGACY_SHORT_COHORT=1 \
  bash scripts/cache.sh
echo 'CONTROL JOB 2/2: existing long MMAU50, forward10, six unhalved schedules'
env MANIFEST="$long_manifest" RUN_NAME="$long_output" ALLOW_LEGACY_SHORT_COHORT=0 \
  bash scripts/cache.sh
archive="${RESULTS_ARCHIVE:-runs/forward10_unhalved_controls.tar.gz}"
python - "$archive" "$original_output" "$long_output" <<'PY'
import sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1]);archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as bundle:
    for folder in sys.argv[2:]:
        path=Path(folder);bundle.add(path,arcname=path.name)
print(f'BOTH UNHALVED CONTROL RUNS COMPLETE: {archive}',flush=True)
PY
