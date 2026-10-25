#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${AUDIO_VENV:-$PWD/.venv}/bin/activate"
export DIFFA_ROOT="${DIFFA_ROOT:-$PWD/external/DIFFA}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
python -m unittest discover -s tests -v
