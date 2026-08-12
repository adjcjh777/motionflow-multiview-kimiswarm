#!/usr/bin/env bash
# A800-D medium run for v83 view-conditioned temporal attention on the
# non-circular H36M true-GT standard protocol (S1,5,6,7,8 train -> S9/S11 val).
#
# This is the v83 ablation: v25 multi-view geometry fusion + view-conditioned
# temporal attention over ray tokens.
#
# Usage
# -----
#   # Default: GPU 4 (override with CUDA_VISIBLE_DEVICES if busy)
#   bash scripts/run_v83_true_gt_h36m_medium_a800.sh
#
#   # Run on a specific free GPU
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_v83_true_gt_h36m_medium_a800.sh

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: default to GPU 4, but allow CUDA_VISIBLE_DEVICES override.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --num_domains 1 \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_view_conditioned_temporal_attention_v83 \
    --v83_d 128 \
    --v83_n_heads 4 \
    --v83_temporal_window 9 \
    --v83_n_layers 1 \
    --v83_dropout 0.1 \
    --v83_residual_gate_init -6.0 \
    --v83_use_view_reliability_bias \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 10 \
    --batch_size 32 \
    --train_samples 64 \
    --val_stride 20 \
    --lr 1e-4 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v83_true_gt_h36m_medium_a800.pth \
    > outputs/ablations/v83_true_gt_h36m_medium_a800.log 2>&1
