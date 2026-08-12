#!/usr/bin/env bash
# Local RTX 4090 smoke for v25 stability ablation.
#
# Tests the hypothesis that a lower learning rate, longer warmup, and disabling
# view permutation prevent the Epoch 2/3 validation MPJPE blow-up observed on
# the true-GT H36M protocol.
#
# This smoke is capped at 5 epochs so it finishes in < 30 min on the local
# RTX 4090. If validation MPJPE stays stable, use the A800 companion script for
# the full 20-epoch run.
#
# Usage:
#   bash scripts/run_v25_ablation_true_gt_stability_local_4090.sh

set -euo pipefail

PYTHON=${PYTHON:-python}

mkdir -p outputs/ablations

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
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
    --num_workers 0 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 5 --batch_size 16 --train_samples 4096 --val_stride 20 \
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
    --output outputs/ablations/v25_true_gt_stability_smoke.pth \
    > outputs/ablations/v25_true_gt_stability_smoke.log 2>&1
