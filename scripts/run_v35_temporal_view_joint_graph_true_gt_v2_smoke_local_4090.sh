#!/usr/bin/env bash
# v35 Temporal View-Joint Graph Network smoke run on local RTX 4090 using the
# corrected (non-circular) H36M true-GT v2 protocol.
#
# Usage
# -----
#   bash scripts/run_v35_temporal_view_joint_graph_true_gt_v2_smoke_local_4090.sh
#
#   # Run on a specific GPU
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_v35_temporal_view_joint_graph_true_gt_v2_smoke_local_4090.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-python}

mkdir -p outputs/smoke_v35_true_gt_v2

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
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
    --use_view_joint_graph_network_v34 \
    --v34_vjgn_n_layers 2 \
    --v34_vjgn_n_heads 4 \
    --use_temporal_view_joint_graph_network_v35 \
    --v35_tvjgn_n_layers 2 \
    --v35_tvjgn_n_heads 4 \
    --num_workers 0 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 2 \
    --batch_size 8 \
    --train_samples 512 \
    --val_stride 10 \
    --lr 1e-4 \
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
    --output outputs/smoke_v35_true_gt_v2/v35_temporal_view_joint_graph_true_gt_v2_smoke_local_4090.pth \
    > outputs/smoke_v35_true_gt_v2/v35_temporal_view_joint_graph_true_gt_v2_smoke_local_4090.log 2>&1
