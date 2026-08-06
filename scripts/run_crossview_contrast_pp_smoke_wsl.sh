#!/usr/bin/env bash
# Smoke test for cross-view contrastive pose representation (Tier-1 iter15 proposal).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_crossview_contrast_pp_smoke_mpiinf3dhp.py
