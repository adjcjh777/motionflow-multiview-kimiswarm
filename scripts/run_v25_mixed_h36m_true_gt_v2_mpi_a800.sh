#!/usr/bin/env bash
# A800 launch script for v25 mixed-dataset training:
# H36M true-GT v2 + MPI-INF-3DHP detected-2D.
#
# This manifest uses the corrected non-circular H36M true-GT v2 labels
# (configs/splits/h36m_true_gt_v2_standard.yaml) and the RTMPose detected-2D
# MPI-INF-3DHP .npz files. MPI-INF-3DHP has 14 views, so the variable-view
# training ceiling is set to 14.
#
# Usage
# -----
#   # Default: run on GPU 6
#   bash scripts/run_v25_mixed_h36m_true_gt_v2_mpi_a800.sh
#
#   # Run on a different GPU (e.g. GPU 7)
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_mixed_h36m_true_gt_v2_mpi_a800.sh
#
#   # Use a different Python interpreter
#   PYTHON=/path/to/python bash scripts/run_v25_mixed_h36m_true_gt_v2_mpi_a800.sh

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: default to GPU 6, but allow CUDA_VISIBLE_DEVICES override.
# GPU 6 is the preferred target for this ablation; check nvidia-smi before launching.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

# Mixed-dataset ablation: H36M true-GT v2 + MPI-INF-3DHP detected-2D.
# Hypothesis: cross-dataset diversity from MPI-INF-3DHP improves robustness.
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_mpi_detected_2d_mixed_train_val_a800.yaml \
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
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v25_mixed_h36m_true_gt_v2_mpi_a800.pth \
    > outputs/ablations/v25_mixed_h36m_true_gt_v2_mpi_a800.log 2>&1
