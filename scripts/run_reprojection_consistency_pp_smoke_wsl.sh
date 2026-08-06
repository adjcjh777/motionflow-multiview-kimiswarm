#!/usr/bin/env bash
# Smoke test for the reprojection-consistency loss (Tier-1 iter14 proposal).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_reprojection_consistency_pp_smoke_mpiinf3dhp.py
