#!/usr/bin/env bash
# The first three examples of the same frozen 50-example cohort.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SAMPLES=3
export RUN_NAME="${RUN_NAME:-runs/smoke3_fpd128}"
exec bash scripts/run.sh
