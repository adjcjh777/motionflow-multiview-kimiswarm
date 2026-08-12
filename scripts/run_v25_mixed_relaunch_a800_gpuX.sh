#!/usr/bin/env bash
# A800 relaunch script for v25 H36M true-GT + AIST++ mixed-dataset training.
#
# This script avoids the divergence seen in the first mixed-dataset attempt
# (outputs/ablations/v25_true_gt_mixed_dataset_a800_gpu5.log) by using the
# conservative "stability" recipe:
#   - lower learning rate (1e-4)
#   - longer cosine warmup (4 epochs)
#   - weight decay (1e-4)
#   - no variable-view permutation
#   - capped variable-view range (max 4 views)
#   - reduced outlier-view augmentation (15%)
#   - early stopping with patience 3
#
# Usage:
#   nohup bash scripts/run_v25_mixed_relaunch_a800_gpuX.sh <GPU_ID> &
# Example:
#   nohup bash scripts/run_v25_mixed_relaunch_a800_gpuX.sh 4 &

set -euo pipefail

# GPU selection: first argument, default to 4 if not provided.
GPU=${1:-4}
export CUDA_VISIBLE_DEVICES=${GPU}

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

# Distinct output names so this relaunch does not overwrite prior runs.
OUT_PREFIX="outputs/ablations/v25_true_gt_mixed_relaunch_a800_gpu${GPU}"

nohup "$PYTHON" -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml \
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
    --num_workers 4 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 1e-4 --lr_cosine --lr_warmup_epochs 4 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output "${OUT_PREFIX}.pth" \
    > "${OUT_PREFIX}.log" 2>&1 &

PID=$!
echo "Launched v25 mixed-dataset relaunch on GPU ${GPU} (PID: ${PID})"
echo "Checkpoint: ${OUT_PREFIX}.pth"
echo "Log:        ${OUT_PREFIX}.log"
