#!/usr/bin/env bash
# CPU smoke test for Bayesian triangulation v2 (batched lstsq DLT).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_bayesian_tri_v2_smoke.py \
    --epochs 2 \
    --batch_size 2 \
    --clip_len 5 \
    --d 16 \
    --n_st_layers 1 \
    --residual_hidden 32
