#!/usr/bin/env bash
# WebBridge data expansion smoke: v25 small with expanded H36M manifest.
# Forces CPU so it can run while the local 4090 v25 baseline is on GPU.
set -euo pipefail

PYTHON=${PYTHON:-/d/anaconda3/python}
OUTPUT=${OUTPUT:-outputs/webbridge_data_expansion_v25_smoke.pth}
LOG=${LOG:-outputs/webbridge_data_expansion_v25_smoke.log}
MANIFEST=configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml

# Use a tiny configuration that completes in <30 min on CPU.
CUDA_VISIBLE_DEVICES=9 $PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest $MANIFEST \
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
    --d 32 --residual_hidden 64 --n_st_layers 1 \
    --graph_num_layers 0 --n_joint_layers 0 --n_heads 2 \
    --epochs 2 --batch_size 4 --train_samples 5 --val_stride 200 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output $OUTPUT \
    > $LOG 2>&1
