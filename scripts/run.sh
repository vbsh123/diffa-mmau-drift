#!/usr/bin/env bash
# Unattended setup + preparation + official-FPD drift + figures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup.sh
study_manifest="${MANIFEST:-$PWD/data/mmau50.json}"
python -u -m drift_study.data --out "$study_manifest" --samples "${COHORT_SAMPLES:-50}"
study_samples="${SAMPLES:-50}"
study_verify="${VERIFY_SAMPLES:-2}"
if (( study_verify > study_samples )); then study_verify="$study_samples"; fi
python -u -m drift_study.run \
  --manifest "$study_manifest" \
  --out "${RUN_NAME:-runs/mmau50_fpd128}" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}" \
  --samples "$study_samples" \
  --verify-samples "$study_verify" \
  --pairwise-domains "${PAIRWISE_DOMAINS:-3}"
