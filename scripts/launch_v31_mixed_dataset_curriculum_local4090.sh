#!/usr/bin/env bash
# v31 mixed-dataset curriculum smoke test on local RTX 4090.
# Three-stage warm-start: H36M-only -> 3:1 H36M:MPI -> 1:1 H36M:MPI.
set -euo pipefail

export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}

OUT_DIR=outputs/v31_mixed_dataset_curriculum
STAGE1_PTH="${OUT_DIR}/stage1_h36m_only.pth"
STAGE2_PTH="${OUT_DIR}/stage2_h36m_mpi_3_1.pth"
STAGE3_PTH="${OUT_DIR}/stage3_h36m_mpi_1_1.pth"

COMMON_FLAGS=(
    --use_mixed_loader
    --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding
    --use_deformable_cross_view_attention_v18
    --use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment
    --num_workers 0 --d 64 --residual_hidden 128 --n_st_layers 2
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4
    --clip_len 9 --batch_size 4 --train_samples 200 --val_stride 10
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999
    --early_stopping_patience 2 --early_stopping_min_delta 0.001
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true
    --use_entropy_regularization true --attention_entropy_weight 0.01
    --use_camera_view_embedding --use_set_view_aggregator
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 8
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 --aleatoric_reproj_loss_weight 0.1
    --outlier_view_prob 0.3 --outlier_view_max_views 1
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0
    --use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1
    --use_physical_space_temporal_loss_v29
    --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001
    --v29_physical_loss_warmup_epochs 1
)

mkdir -p "${OUT_DIR}"

echo "[v31 curriculum] Stage 1: H36M-only smoke..."
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    "${COMMON_FLAGS[@]}" \
    --mixed_manifest configs/v31_mixed_dataset_curriculum/stage1_h36m_only_smoke.yaml \
    --epochs 2 \
    --output "${STAGE1_PTH}" \
    > "${OUT_DIR}/stage1_h36m_only.log" 2>&1

echo "[v31 curriculum] Stage 2: 3:1 H36M:MPI, warm-started from stage 1..."
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    "${COMMON_FLAGS[@]}" \
    --mixed_manifest configs/v31_mixed_dataset_curriculum/stage2_h36m_mpi_3_1_smoke.yaml \
    --warm_start "${STAGE1_PTH}" \
    --epochs 2 \
    --output "${STAGE2_PTH}" \
    > "${OUT_DIR}/stage2_h36m_mpi_3_1.log" 2>&1

echo "[v31 curriculum] Stage 3: 1:1 H36M:MPI, warm-started from stage 2..."
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    "${COMMON_FLAGS[@]}" \
    --mixed_manifest configs/v31_mixed_dataset_curriculum/stage3_h36m_mpi_1_1_smoke.yaml \
    --warm_start "${STAGE2_PTH}" \
    --epochs 2 \
    --output "${STAGE3_PTH}" \
    > "${OUT_DIR}/stage3_h36m_mpi_1_1.log" 2>&1

echo "[v31 curriculum] Smoke test complete. Final checkpoint: ${STAGE3_PTH}"
