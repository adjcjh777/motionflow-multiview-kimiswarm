#!/usr/bin/env bash
# v29: v26 + UDP-GMM + v28 + camera-joint graph + outlier-view detector.
# More complex multi-view fusion with explicit robustness components.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
OUTPUT=${OUTPUT:-outputs/omniview_fusion_v29_udp_gmm_v28_graph_outlier_full_local_4090.pth}
LOG=${LOG:-outputs/omniview_fusion_v29_udp_gmm_v28_graph_outlier_full_local_4090.log}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_temporal_geometry_fusion_v26 \
    --v26_temporal_window 3 \
    --use_uncertainty_depth_proposals_v27 \
    --v27_uncertainty_loss_weight 0.01 \
    --v27_udp_n_mixtures 2 \
    --use_physical_space_alignment_v28 \
    --v28_floor_loss_weight 0.01 --v28_bone_temporal_weight 0.01 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --v25_use_camera_joint_graph \
    --v25_use_outlier_view_detector \
    --num_workers 0 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 50 --batch_size 16 --train_samples 500 --val_stride 10 \
    --lr 3e-4 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 3 --early_stopping_min_delta 1e-5 \
    --weight_decay 5e-5 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output $OUTPUT \
    > $LOG 2>&1
