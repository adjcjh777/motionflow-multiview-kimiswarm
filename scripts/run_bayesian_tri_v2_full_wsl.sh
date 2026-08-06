#!/usr/bin/env bash
# Full training for Bayesian triangulation v2 (batched lstsq DLT).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_bayesian_tri_v2_full_mpiinf3dhp.py
