#!/usr/bin/env bash
# v31 geometry-aware camera view embedding smoke test on local RTX 4090.
set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --num_workers 0 --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --clip_len 9 --epochs 2 --batch_size 4 --train_samples 50 --val_stride 1 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 2 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding_v31 --camera_view_embedding_hidden 32 --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 8 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 2 \
    --output outputs/omniview_fusion_v31_camera_view_embedding_smoke_local_4090.pth \
    > outputs/omniview_fusion_v31_camera_view_embedding_smoke_local_4090.log 2>&1 &
echo $! > outputs/v31_camera_view_embedding_smoke_local_4090.pid
wait $!
