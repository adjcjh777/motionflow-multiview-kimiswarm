#!/usr/bin/env bash
# A800-D long-horizon run for v80 (view-reliability weighting) on the
# non-circular true-GT Shelf/Campus detected protocol.
#
# Same model/loss configuration as the verified 3-epoch smoke
# (outputs/omniview_fusion_v80_shelf_campus_detected_smoke.config.json),
# scaled to 25 epochs so the model can actually converge and try to close
# the gap to the ~122 mm root-aligned DLT baseline
# (docs/results_true_gt_shelf_campus.md).
#
# GPU discipline (project rule): at most 2 GPUs, fixed indices via
# CUDA_VISIBLE_DEVICES. Default 4,5 (verified free 2026-08-10); override
# with GPUS=<a,b> if occupancy changed. Check nvidia-smi first.
set -euo pipefail

GPUS=${GPUS:-4,5}
export CUDA_VISIBLE_DEVICES="$GPUS"

PYTHON=${PYTHON:-python}
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/shelf_campus_detected_smoke.yaml \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 \
    --v46_svg_use_curriculum \
    --use_v80_view_reliability \
    --v80_vrbt_hidden 64 --v80_vrbt_n_layers 2 \
    --v80_vrbt_weight_type per_view_joint \
    --v80_vrbt_use_geometry_bias --v80_vrbt_use_feature_bias \
    --v80_vrbt_identity_init --v80_vrbt_min_weight 0.05 \
    --num_workers 0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 25 --batch_size 8 --train_samples 512 --val_stride 2 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 8 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 2 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/omniview_fusion_v80_shelf_campus_detected_long.pth \
    > outputs/omniview_fusion_v80_shelf_campus_detected_long.log 2>&1
