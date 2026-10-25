#!/usr/bin/env bash
# Unattended setup + preparation + official-FPD drift + figures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup.sh
study_manifest="${MANIFEST:-$PWD/data/mmau50_long.json}"
audio_duration_args=(--min-seconds "${MIN_SECONDS:-60}")
if [[ -n "${MAX_SECONDS:-}" ]]; then audio_duration_args+=(--max-seconds "$MAX_SECONDS"); fi
python -u -m drift_study.data --out "$study_manifest" --samples "${COHORT_SAMPLES:-50}" "${audio_duration_args[@]}"
study_samples="${SAMPLES:-50}"
study_verify="${VERIFY_SAMPLES:-2}"
if (( study_verify > study_samples )); then study_verify="$study_samples"; fi
python -u -m drift_study.run \
  --manifest "$study_manifest" \
  --out "${RUN_NAME:-runs/mmau50_long_fpd128}" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --samples "$study_samples" \
  --verify-samples "$study_verify" \
  --pairwise-domains "${PAIRWISE_DOMAINS:-3}" \
  --attention-every "${ATTENTION_EVERY:-1}" \
  --attention-query-chunk "${ATTENTION_QUERY_CHUNK:-128}"
