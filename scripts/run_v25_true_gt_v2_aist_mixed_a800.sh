#!/usr/bin/env bash
# A800 launch script for v25 H36M true-GT v2 + AIST++ mixed-dataset training.
#
# This is the v2-label counterpart of scripts/run_v25_ablation_mixed_dataset_a800.sh.
# It uses the corrected non-circular H36M true-GT v2 manifest and keeps the same
# v25 hyperparameters as the original mixed-dataset ablation.
#
# Usage
# -----
#   # Default: run on GPU 6
#   nohup bash scripts/run_v25_true_gt_v2_aist_mixed_a800.sh &
#
#   # Run on GPU 7
#   nohup bash scripts/run_v25_true_gt_v2_aist_mixed_a800.sh 7 &
#
# IMPORTANT: This project may only use GPUs 6 and 7. Do not pass 0-5.

set -euo pipefail

# GPU selection: default to 6, allow override by first positional argument.
GPU=${1:-6}
if [[ "$GPU" != "6" && "$GPU" != "7" ]]; then
    echo "Error: this project may only use GPUs 6 or 7 (got GPU $GPU)" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

OUT_PREFIX="outputs/ablations/v25_true_gt_v2_mixed_dataset_a800_gpu${GPU}"

nohup "$PYTHON" -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_aist_mixed_train_val_a800.yaml \
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
    --lr 5e-4 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 9 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output "${OUT_PREFIX}.pth" \
    > "${OUT_PREFIX}.log" 2>&1 &

PID=$!
echo "Launched v25 true-GT v2 + AIST++ mixed-dataset training on GPU ${GPU} (PID: ${PID})"
echo "Checkpoint: ${OUT_PREFIX}.pth"
echo "Log:        ${OUT_PREFIX}.log"
