#!/usr/bin/env bash
# Separate checkout/job: wait for drift to complete before running on the same GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${BASELINE_RUN:-}" ]]; then
  echo 'Set BASELINE_RUN to the completed 50-example drift run (not its analysis/ folder).'
  exit 2
fi
source scripts/setup.sh
study_manifest="${MANIFEST:-$PWD/data/mmau50_long.json}"
audio_duration_args=(--min-seconds "${MIN_SECONDS:-60}")
if [[ -n "${MAX_SECONDS:-}" ]]; then audio_duration_args+=(--max-seconds "$MAX_SECONDS"); fi
python -u -m drift_study.data --out "$study_manifest" --samples 50 "${audio_duration_args[@]}"
cache_reference_args=()
if [[ -n "${BASELINE_MANIFEST:-}" ]]; then
  cache_reference_args+=(--baseline-manifest "$BASELINE_MANIFEST")
fi
cache_samples="${SAMPLES:-50}"
cache_verify="${VERIFY_SAMPLES:-2}"
if (( cache_verify > cache_samples )); then cache_verify="$cache_samples"; fi
cache_start="${CACHE_START:-mean}"
cache_layers="${CACHE_LAYERS:-all}"
python -u -m cache_study.run \
  --manifest "$study_manifest" \
  --baseline-run "$BASELINE_RUN" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --out "${RUN_NAME:-runs/cache50_long_${cache_start}_${cache_layers//,/_}}" \
  --samples "$cache_samples" --verify-samples "$cache_verify" \
  --start "$cache_start" --cache-layers "$cache_layers" \
  "${cache_reference_args[@]}"
