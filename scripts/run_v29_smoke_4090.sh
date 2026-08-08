#!/usr/bin/env bash
# v29 CPU/GPU smoke test on RTX 4090.
set -euo pipefail

export PYTHONUNBUFFERED=1

PYTHON=${PYTHON:-python}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_hierarchical_multiview_v29 \
    --use_physical_space_temporal_loss_v29 \
    --num_workers 0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 1 --batch_size 2 --train_samples 4 --val_stride 1 \
    --lr 1e-3 --max_grad_norm 1.0 \
    --output outputs/v29_smoke_local_4090.pth \
    > outputs/v29_smoke_local_4090.log 2>&1

echo "v29 smoke test completed; see outputs/v29_smoke_local_4090.log"
