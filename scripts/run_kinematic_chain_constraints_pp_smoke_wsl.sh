#!/usr/bin/env bash
# Smoke test for the kinematic-chain constraints auxiliary loss.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_kinematic_chain_constraints_pp_smoke_mpiinf3dhp.py
