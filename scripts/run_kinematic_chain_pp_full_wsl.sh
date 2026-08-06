#!/usr/bin/env bash
# Full training for the kinematic-chain graph refiner (Tier-1 iter15 proposal).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_kinematic_chain_pp_full_mpiinf3dhp.py
