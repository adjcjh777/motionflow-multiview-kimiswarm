#!/usr/bin/env bash
# v49 ablation matrix on the local RTX 4090.
# Runs a small smoke for each variant so we can compare v45/v46/v47/v48/v49-Lite.
# Intended to be launched once the GPU is free (do not overlap with other training).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_FLAGS=(
    --use_mixed_loader
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml
    --use_domain_embedding
    --use_deformable_cross_view_attention_v18
    --use_multiview_geometry_fusion_v25
    --v25_geom_loss_weight 0.1
    --v25_dropout 0.2
    --v25_use_geometry_attention
    --v25_use_learned_depth_triangulation
    --v25_use_geometry_bundle_adjustment
    --use_v45_adaptive_geometry_fusion
    --v45_adaptive_weight_type per_view_joint
    --v45_adaptive_weight_hidden 32
    --v45_adaptive_weight_n_layers 1
    --num_workers 0
    --d 64
    --residual_hidden 128
    --n_st_layers 2
    --graph_num_layers 1
    --n_joint_layers 1
    --n_heads 4
    --clip_len 9
    --epochs 2
    --batch_size 4
    --train_samples 500
    --val_stride 10
    --lr 1e-3
    --lr_cosine
    --lr_warmup_epochs 1
    --lr_min 1e-6
    --max_grad_norm 1.0
    --ema_decay 0.999
    --early_stopping_patience 2
    --early_stopping_min_delta 0.001
    --use_multiscale_fusion true
    --use_camera_conditioning true
    --use_epipolar_bias true
    --use_context_visibility true
    --use_skeleton_residual true
    --use_rotation_correction true
    --use_entropy_regularization true
    --attention_entropy_weight 0.01
    --use_camera_view_embedding
    --use_set_view_aggregator
    --pa_loss_weight 0.5
    --monotonic_loss_weight 0.1
    --monotonic_margin 5.0
    --reproj_loss_weight 0.1
    --reproj_warmup_epochs 1
    --aleatoric_reproj_loss_weight 0.1
)

run_variant() {
    local name="$1"
    shift
    local extra_flags=("$@")
    echo "$(date -Iseconds) starting ${name}"
    python -u "${SCRIPT_DIR}/../experiments/train_omniview_fusion_v5_webbridge_multi.py" \
        "${COMMON_FLAGS[@]}" \
        "${extra_flags[@]}" \
        --output "outputs/v49_ablation_${name}.pth" \
        >> "outputs/v49_ablation_${name}.log" 2>&1
    echo "$(date -Iseconds) ${name} done"
    sleep 120
}

echo "$(date -Iseconds) starting v49 ablation matrix on local RTX 4090"

# 1. v45 baseline (adaptive geometry fusion only).
run_variant "v45_only"

# 2. v45 + v46 sparse-view generalization.
run_variant "v46_on_v45" \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum

# 3. v45 + v46 + v47 temporal aggregation.
run_variant "v47_on_v46" \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v47_temporal_aggregation \
    --v47_temporal_d_model 64 \
    --v47_temporal_n_heads 4 \
    --v47_temporal_num_layers 2 \
    --v47_temporal_loss_weight 0.01 \
    --v47_use_view_count_conditioning

# 4. v45 + v46 + v47 + v48 domain generalization.
run_variant "v48_on_v47" \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v47_temporal_aggregation \
    --v47_temporal_d_model 64 \
    --v47_temporal_n_heads 4 \
    --v47_temporal_num_layers 2 \
    --v47_temporal_loss_weight 0.01 \
    --v47_use_view_count_conditioning \
    --use_v48_domain_generalization \
    --v48_dg_hidden 64 \
    --v48_dg_grl_lambda 0.1 \
    --v48_dg_use_domain_film \
    --v48_dg_use_ddwl \
    --v48_dg_ddwl_temperature 2.0 \
    --v48_dg_ddwl_warmup_epochs 1

# 5. v45 + v46 + v49-Lite + v48 (v47 replaced by v49-Lite).
run_variant "v49_lite_on_v46" \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v49_lite_temporal_aggregation \
    --v49_lite_temporal_d_model 32 \
    --v49_lite_temporal_num_layers 2 \
    --v49_lite_temporal_kernel_size 3 \
    --v49_lite_temporal_loss_weight 0.01 \
    --v49_lite_temporal_use_view_count_conditioning \
    --use_v48_domain_generalization \
    --v48_dg_hidden 64 \
    --v48_dg_grl_lambda 0.1 \
    --v48_dg_use_domain_film \
    --v48_dg_use_ddwl \
    --v48_dg_ddwl_temperature 2.0 \
    --v48_dg_ddwl_warmup_epochs 1

echo "$(date -Iseconds) v49 ablation matrix complete"
