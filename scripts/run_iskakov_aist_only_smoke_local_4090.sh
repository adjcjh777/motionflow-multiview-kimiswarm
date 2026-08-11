#!/usr/bin/env bash
# Local RTX 4090 AIST++ only non-circular smoke for Iskakov ICCV 2019 learnable triangulation.
# AIST++ uses 9 views and 17 joints (same skeleton as H36M).
set -euo pipefail

PYTHON=${PYTHON:-python}
$PYTHON -u experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol aist_smoke \
    --epochs 30 \
    --batch_size 4 \
    --lr 1e-3 \
    --hidden_dim 32 \
    --train_samples_per_epoch 128 \
    --log_path outputs/iskakov_learnable_tri_aist_only_smoke.log \
    --ckpt_path outputs/iskakov_learnable_tri_aist_only_smoke.pth
