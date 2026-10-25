#!/usr/bin/env bash
# Install the environment and fetch source only; does not run model inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
study_venv="${AUDIO_VENV:-$PWD/.venv}"
if [[ ! -x "$study_venv/bin/python" ]]; then
  "${PYTHON:-python3}" -m venv "$study_venv"
fi
source "$study_venv/bin/activate"
python -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10–3.12"'
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r requirements.txt
  python -m pip check
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
study_diffa="${DIFFA_ROOT:-$PWD/external/DIFFA}"
python -m drift_study.checkout --diffa-root "$study_diffa"
echo 'SETUP COMPLETE'
