#!/usr/bin/env bash
# Local RTX 4090: Iskakov ICCV 2019 on the full AIST++ train/val split.
# 1280 train clips, 128 val clips, 9 views, 17 joints.
set -euo pipefail

PYTHON=${PYTHON:-python}
$PYTHON -u experiments/train_iskakov_aistpp_full.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --patience 3 \
    --seed 20260811 \
    --log_path outputs/iskakov_learnable_tri_aistpp_full.log \
    --ckpt_path outputs/iskakov_learnable_tri_aistpp_full.pth
