#!/usr/bin/env bash
# One GPU: old-cohort caching, long-cohort caching, then long drift + attention.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# This launcher is the requested full experiment, independent of stale shell overrides.
unset MAX_SECONDS BASELINE_RUN BASELINE_MANIFEST
export SAMPLES=50 COHORT_SAMPLES=50 MIN_SECONDS=60 CACHE_LAYERS=all
export VERIFY_SAMPLES=2 MATCHED_FLAT=0 PAIRWISE_DOMAINS=3
export ATTENTION_EVERY=1 ATTENTION_QUERY_CHUNK=128
source scripts/setup.sh
export SKIP_INSTALL=1
export AUDIO_VENV="$study_venv"
export DIFFA_ROOT="$study_diffa"
original_manifest="${ORIGINAL_MANIFEST:-/workspace/diffa-mmau-drift/data/mmau50.json}"
long_manifest="${LONG_MANIFEST:-$PWD/data/mmau50_long.json}"
original_output="${ORIGINAL_CACHE_RUN_NAME:-runs/cache50_original_forward10_all_deepsem2_24-31}"
long_output="${LONG_CACHE_RUN_NAME:-runs/cache50_long_forward10_all_deepsem2_24-31}"
drift_output="${DRIFT_RUN_NAME:-runs/mmau50_long_fpd128_attention}"
# Fail before model loading if the original cohort cannot be found/verified.
python - "$original_manifest" "$long_manifest" "$original_output" "$long_output" "$drift_output" <<'PY'
import sys
from pathlib import Path
from drift_study.data import read_manifest
paths=[Path(item).resolve() for item in sys.argv[1:]]
if paths[0]==paths[1]:
    raise ValueError('Original and long cohorts need separate manifests')
outputs=paths[2:]
if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]):
    raise ValueError('All three output directories must be separate and non-nested')
manifest=read_manifest(sys.argv[1])
if len(manifest['rows'])!=50:
    raise ValueError('Need the existing frozen original 50-question manifest')
print('Original cohort: all 50 WAV hashes and prompts verified',flush=True)
PY
echo "JOB 1/3: original MMAU50 forward-10 layerwise caching"
env -u BASELINE_RUN -u BASELINE_MANIFEST MANIFEST="$original_manifest" RUN_NAME="$original_output" \
  CACHE_START=fixed CACHE_START_FORWARD=10 DEEP_SEMANTIC_FACTOR=2 \
  DEEP_SEMANTIC_LAYERS=24-31 ALLOW_LEGACY_SHORT_COHORT=1 bash scripts/cache.sh
echo "JOB 2/3: new long MMAU50 forward-10 layerwise caching"
env -u BASELINE_RUN -u BASELINE_MANIFEST MANIFEST="$long_manifest" RUN_NAME="$long_output" \
  CACHE_START=fixed CACHE_START_FORWARD=10 DEEP_SEMANTIC_FACTOR=2 \
  DEEP_SEMANTIC_LAYERS=24-31 ALLOW_LEGACY_SHORT_COHORT=0 bash scripts/cache.sh
echo "JOB 3/3: new long MMAU50 drift + attention (separate measurement run)"
env MANIFEST="$long_manifest" RUN_NAME="$drift_output" CACHE_REFERENCE_RUN="$long_output" \
  bash scripts/run.sh
archive="${RESULTS_ARCHIVE:-runs/next_results.tar.gz}"
python - "$archive" "$original_output" "$long_output" "$drift_output" <<'PY'
import sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1]); archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as bundle:
    for item in sys.argv[2:]:
        path=Path(item)
        bundle.add(path,arcname=path.name)
print(f'ALL THREE JOBS COMPLETE: {archive}')
PY
