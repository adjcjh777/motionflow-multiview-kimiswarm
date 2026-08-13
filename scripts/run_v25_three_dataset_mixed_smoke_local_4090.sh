#!/usr/bin/env bash
# Local RTX 4090 smoke for v25 baseline on H36M true-GT v2 + AIST++ +
# MPI-INF-3DHP detected-2D mixed training.
#
# Usage
# -----
#   bash scripts/run_v25_three_dataset_mixed_smoke_local_4090.sh
#
#   # Force a specific local GPU (default: GPU 0, never A800 GPU 6/7)
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_v25_three_dataset_mixed_smoke_local_4090.sh
set -euo pipefail

# Pin to the local RTX 4090 (single GPU). This script must NOT touch A800 GPUs 6/7.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-python}

mkdir -p outputs

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_aist_mpi_mixed_train_val_a800.yaml \
    --num_domains 3 \
    --use_domain_embedding \
    --domain_loss_weights "1.0,0.2,1.0" \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --num_workers 0 \
    --d 64 \
    --residual_hidden 128 \
    --n_st_layers 2 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 9 \
    --epochs 2 \
    --batch_size 4 \
    --train_samples 256 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
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
    --output outputs/omniview_fusion_v25_three_dataset_mixed_smoke_local_4090.pth \
    > outputs/omniview_fusion_v25_three_dataset_mixed_smoke_local_4090.log 2>&1
