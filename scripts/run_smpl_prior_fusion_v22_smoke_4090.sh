#!/usr/bin/env bash
# v22 SMPL prior fusion local 4090 smoke test.
set -euo pipefail

PYTHON=${PYTHON:-python}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_smpl_prior_fusion_v22 \
    --smpl_prior_loss_weight 0.1 \
    --epochs 1 \
    --batch_size 2 \
    --d 32 \
    --residual_hidden 64 \
    --n_st_layers 1 \
    --graph_num_layers 1 \
    --num_workers 0 \
    --output outputs/smpl_prior_fusion_v22_smoke.pth

echo "==> v22 SMPL prior fusion smoke test passed"
