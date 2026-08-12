#!/usr/bin/env bash
# A800-D medium run for v86 Stage 2: mixed H36M true-GT + AIST++ cross-domain
# training with the unified v86 sparse-view stack.
#
# Usage
# -----
#   # Default: GPU 7 (project only uses GPUs 6/7; override if busy)
#   bash scripts/run_v86_sparse_cross_domain_mixed_medium_a800.sh
#
#   # Run on a specific free GPU
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_v86_sparse_cross_domain_mixed_medium_a800.sh

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: default to GPU 7, but allow CUDA_VISIBLE_DEVICES override.
# Project policy permits only GPU 6/7; GPU 0-5 are reserved.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

# Optional warm-start from Stage 1 (single-domain v86) to preserve sparse-view
# behaviour before mixing in AIST++.  Uncomment the following line if the Stage 1
# checkpoint exists:
# WARM_START="--warm_start outputs/ablations/v86_sparse_cross_domain_medium_a800.pth"

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml \
    --num_domains 2 \
    --use_full_precision_dlt \
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
    --use_random_view_dropout_v85 \
    --v85_dropout_prob 0.3 \
    --v85_min_views 2 \
    --v85_use_count_embedding \
    --use_v86_strong_count_conditioning \
    --v86_count_hidden 64 \
    --v86_count_n_layers 2 \
    --v86_count_dropout 0.1 \
    --use_v86_separate_sparse_view_head \
    --v86_ssv_head_hidden 128 \
    --v86_ssv_head_n_layers 2 \
    --v86_ssv_head_dropout 0.1 \
    --v86_ssv_head_use_count_embedding \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 20 \
    --batch_size 16 \
    --train_samples 4096 \
    --val_stride 20 \
    --lr 1e-4 \
    --lr_cosine \
    --lr_warmup_epochs 4 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --weight_decay 1e-4 \
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
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v86_sparse_cross_domain_mixed_medium_a800.pth \
    > outputs/ablations/v86_sparse_cross_domain_mixed_medium_a800.log 2>&1
