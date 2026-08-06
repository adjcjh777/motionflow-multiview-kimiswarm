#!/usr/bin/env bash
# Full training for uncertainty-aware multi-view triangulation (Tier-2 iter15 proposal).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_bayesian_tri_pp_full_mpiinf3dhp.py
