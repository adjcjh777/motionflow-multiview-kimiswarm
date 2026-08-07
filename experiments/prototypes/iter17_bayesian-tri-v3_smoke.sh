#!/usr/bin/env bash
# iter17 CPU-only smoke test for Bayesian triangulation v3.
# Reuses the existing synthetic smoke training script with tiny dimensions:
#   d=32, batch_size=2, clip_len=9, epochs=2
# All training is forced to CPU-only.
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV=${MF_VENV:-$(pwd)/.venv}
if [ -f "$VENV/bin/activate" ]; then
    . "$VENV/bin/activate"
fi

mkdir -p outputs

python -u experiments/prototypes/train_bayesian_tri_v3_smoke.py \
    --epochs 2 \
    --batch_size 2 \
    --clip_len 9 \
    --d 32 \
    --n_st_layers 1 \
    --residual_hidden 64 \
    --joint_precision_hidden 64 \
    --refinement_hidden 64 \
    --n_refinement_iters 2 \
    --gn_iters 2 \
    --lr 1e-3 \
    --seed 42 \
    --device cpu \
    --output outputs/iter17_bayesian_tri_v3_smoke.pth
