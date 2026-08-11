#!/usr/bin/env bash
# Local RTX 4090 medium run for v25 (geometry fusion) on the mixed-dataset
# manifest configs/splits/mix_true_gt_v2.yaml (H36M true-GT + AIST++ + Shelf/Campus).
#
# This script guards against launching when another training run appears to be
# active.  The project rule is at most ONE training/GPU task at a time on the
# local RTX 4090.
#
# Mixed-dataset anchors:
#   - H36M true-GT: S1,S5-S8 train -> S9,S11 val (non-circular)
#   - AIST++: genre-based train/val split (train gBR,gHO,gJB,gJS,gKR,gLH;
#             val gLO,gMH,gPO,gWA)
#   - Shelf/Campus: detected train/val split
#
# Expected usage (run only when GPU is free):
#   bash scripts/run_v25_mix_true_gt_v2_medium_local_4090.sh
set -euo pipefail

# GPU guard: refuse to start if another training run is already active.
RUNNING=$(ps aux | grep "experiments/train_omniview_fusion_v5_webbridge_multi.py" | grep -v grep || true)
if [ -n "$RUNNING" ]; then
    echo "GPU appears busy with another training run:" >&2
    echo "$RUNNING" >&2
    echo "Aborting; run only when the GPU is free." >&2
    exit 1
fi

PYTHON=${PYTHON:-python}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/mix_true_gt_v2.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --num_workers 0 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 8 --batch_size 16 --train_samples 1024 --val_stride 20 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/omniview_fusion_v25_mix_true_gt_v2_medium.pth \
    > outputs/omniview_fusion_v25_mix_true_gt_v2_medium.log 2>&1
