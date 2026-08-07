#!/usr/bin/env bash
# CPU-only smoke test for iter17 EMA checkpoint save/load.
set -euo pipefail
cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

python -u experiments/prototypes/iter17_ema-checkpoint-save-load_smoke.py \
    --epochs 2 \
    --batch_size 2 \
    --clip_len 9 \
    --d 32 \
    --n_st_layers 1 \
    --residual_hidden 32 \
    --device cpu
