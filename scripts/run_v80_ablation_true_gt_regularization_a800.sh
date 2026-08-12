#!/usr/bin/env bash
# A800-D regularisation ablation for v80 on the non-circular H36M true-GT
# standard protocol (S1,5,6,7,8 train -> S9/S11 val; issue #194).
#
# This script realises the recipe in
# configs/ablations/v80_true_gt_regularization_a800.yaml.
#
# Background on the overfit being targeted
# ----------------------------------------
# The local RTX 4090 medium run (scripts/run_v80_h36m_true_gt_medium.sh) hit a
# valley of 39.98 mm at epoch 4, then overfit to 133.71 mm by epoch 8.  Earlier
# A800 v80 recipes (v1-v4) showed the same post-epoch-2 collapse.  This ablation
# combines:
#   * more unique samples per epoch (4096 vs 1024)
#   * weight decay 2e-4 (was 0.0 in the medium run)
#   * early stopping with patience 3 and min delta 0.001
#   * lower lr (5e-4) and longer warmup (2 epochs)
#   * milder augmentation (outlier_view_prob 0.15)
#   * explicit geometry regularisation: bone / joint-limit / temporal-bone losses
#   * reduced v25 geometry loss weight and stronger v25 dropout
#
# Sanity anchors
# --------------
# DLT (conf-weighted) true-GT H36M: 25.67 mm
# Iskakov ICCV 2019 true-GT H36M: 23.35 mm
# v80 local medium best: 39.98 mm @ epoch 4
#
# Usage
# -----
#   # Default: run on GPU 4
#   bash scripts/run_v80_ablation_true_gt_regularization_a800.sh
#
#   # Run on GPU 6 (or any other free GPU)
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_v80_ablation_true_gt_regularization_a800.sh
#
#   # Use a different Python interpreter
#   PYTHON=/path/to/python bash scripts/run_v80_ablation_true_gt_regularization_a800.sh

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: default to GPU 4, but allow CUDA_VISIBLE_DEVICES override.
# GPU 4/6 are the preferred queue targets. Check nvidia-smi before launching.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_dropout 0.1 \
    --v50_sefh_identity_init_gate \
    --v50_sefh_loss_weight 0.0 --v50_sefh_aleatoric_weight 0.0 \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 --v51_cdsvr_num_heads 4 --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_use_domain_label --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 --v52_uwt_n_layers 2 --v52_uwt_weight_type per_view_joint \
    --v52_uwt_use_geometry_bias --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init --v52_uwt_min_weight 0.05 --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 0.0001 \
    --use_v80_view_reliability \
    --v80_vrbt_hidden 64 --v80_vrbt_n_layers 2 \
    --v80_vrbt_weight_type per_view_joint \
    --v80_vrbt_use_geometry_bias --v80_vrbt_use_feature_bias \
    --v80_vrbt_identity_init --v80_vrbt_min_weight 0.05 \
    --bone_loss_weight 0.05 \
    --joint_limit_weight 0.01 \
    --temporal_bone_weight 0.005 \
    --num_workers 4 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 5e-4 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 2e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v80_true_gt_regularization_a800.pth \
    > outputs/ablations/v80_true_gt_regularization_a800.log 2>&1
