#!/usr/bin/env bash
# Clean cache benchmark; run sequentially with instrumented measurements.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cache_start="${CACHE_START:-fixed}"
if [[ "$cache_start" != fixed && -z "${BASELINE_RUN:-}" ]]; then
  echo 'Set BASELINE_RUN to the completed 50-example drift run (not its analysis/ folder).'
  exit 2
fi
source scripts/setup.sh
study_manifest="${MANIFEST:-$PWD/data/mmau50_long.json}"
audio_duration_args=(--min-seconds "${MIN_SECONDS:-60}")
if [[ -n "${MAX_SECONDS:-}" ]]; then audio_duration_args+=(--max-seconds "$MAX_SECONDS"); fi
cache_cohort_args=()
if [[ "${ALLOW_LEGACY_SHORT_COHORT:-0}" == 1 ]]; then
  [[ -f "$study_manifest" ]] || { echo "Missing frozen legacy manifest: $study_manifest"; exit 2; }
  cache_cohort_args+=(--allow-legacy-short-cohort)
else
  python -u -m drift_study.data --out "$study_manifest" --samples 50 "${audio_duration_args[@]}"
fi
cache_reference_args=()
if [[ -n "${BASELINE_RUN:-}" ]]; then cache_reference_args+=(--baseline-run "$BASELINE_RUN"); fi
if [[ -n "${BASELINE_MANIFEST:-}" ]]; then
  cache_reference_args+=(--baseline-manifest "$BASELINE_MANIFEST")
fi
cache_samples="${SAMPLES:-50}"
cache_verify="${VERIFY_SAMPLES:-2}"
if (( cache_verify > cache_samples )); then cache_verify="$cache_samples"; fi
cache_layers="${CACHE_LAYERS:-all}"
cache_fast_layers="${DEEP_SEMANTIC_LAYERS:-24-31}"
cache_start_name="$cache_start"
if [[ "$cache_start" == fixed ]]; then cache_start_name="forward${CACHE_START_FORWARD:-10}"; fi
cache_extra_args=()
cache_pair_tag=""
if [[ "${MATCHED_FLAT:-0}" == 1 ]]; then cache_extra_args+=(--matched-flat); cache_pair_tag="_paired"; fi
python -u -m cache_study.run \
  --manifest "$study_manifest" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --out "${RUN_NAME:-runs/cache50_long_${cache_start_name}_${cache_layers//,/_}_deepsem${DEEP_SEMANTIC_FACTOR:-2}_${cache_fast_layers//,/_}${cache_pair_tag}}" \
  --samples "$cache_samples" --verify-samples "$cache_verify" \
  --start "$cache_start" --cache-layers "$cache_layers" \
  --start-forward "${CACHE_START_FORWARD:-10}" \
  --deep-semantic-factor "${DEEP_SEMANTIC_FACTOR:-2}" \
  --deep-semantic-layers "$cache_fast_layers" \
  "${cache_reference_args[@]}" "${cache_extra_args[@]}" "${cache_cohort_args[@]}"
