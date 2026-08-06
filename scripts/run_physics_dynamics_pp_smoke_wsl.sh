#!/usr/bin/env bash
# CPU smoke run for the physics-informed skeleton dynamics prior.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p tmp

python -u experiments/train_physics_dynamics_prior.py \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 --epochs 2 \
  --batch_size 4 --train_samples 64 --val_stride 20 \
  --physics_loss_weight 0.01 \
  --output outputs/physics_dynamics_pp_smoke.pth \
  "$@"
