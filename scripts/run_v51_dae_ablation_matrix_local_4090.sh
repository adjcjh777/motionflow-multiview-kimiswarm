#!/usr/bin/env bash
# v51 DAE ablation matrix on the local RTX 4090.
# Runs small 2-epoch smokes for different v51 DAE hyperparameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_FLAGS=(
    --use_mixed_loader
    --mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml
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
    --use_v46_sparse_view_generalization
    --v46_svg_view_dropout_prob 0.3
    --v46_svg_min_views 2
    --v46_svg_hidden 64
    --v46_svg_use_curriculum
    --use_v51_domain_agnostic_ensemble
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
    --use_variable_view_training
    --variable_view_min_views 2
    --variable_view_max_views 8
    --variable_view_max_views_start 4
    --variable_view_curriculum_alpha 2.0
    --variable_view_permute
    --pa_loss_weight 0.5
    --monotonic_loss_weight 0.1
    --monotonic_margin 5.0
    --reproj_loss_weight 0.1
    --reproj_warmup_epochs 1
    --aleatoric_reproj_loss_weight 0.1
    --outlier_view_prob 0.3
    --outlier_view_max_views 1
    --outlier_view_offset_std 10.0
    --outlier_view_noise_std 15.0
    --use_hierarchical_multiview_v30
    --v30_n_part_layers 2
    --v30_stochastic_depth_prob 0.1
)

run_variant() {
    local name="$1"
    shift
    local extra_flags=("$@")
    echo "$(date -Iseconds) starting ${name}"
    python -u "${SCRIPT_DIR}/../experiments/train_omniview_fusion_v5_webbridge_multi.py" \
        "${COMMON_FLAGS[@]}" \
        "${extra_flags[@]}" \
        --output "outputs/v51_dae_ablation_${name}.pth" \
        >> "outputs/v51_dae_ablation_${name}.log" 2>&1
    echo "$(date -Iseconds) ${name} done"
    sleep 120
}

echo "$(date -Iseconds) starting v51 DAE ablation matrix on local RTX 4090"

# 1. Default v51 DAE (baseline).
run_variant "default" \
    --v51_dae_hidden 64 \
    --v51_dae_num_layers 2 \
    --v51_dae_dropout 0.1 \
    --v51_dae_n_experts 2 \
    --v51_dae_identity_bypass \
    --v51_dae_min_weight 0.05 \
    --v51_dae_loss_weight 0.005

# 2. Lower auxiliary loss weight.
run_variant "low_loss" \
    --v51_dae_hidden 64 \
    --v51_dae_num_layers 2 \
    --v51_dae_dropout 0.1 \
    --v51_dae_n_experts 2 \
    --v51_dae_identity_bypass \
    --v51_dae_min_weight 0.05 \
    --v51_dae_loss_weight 0.001

# 3. No identity bypass (hard switching).
run_variant "no_bypass" \
    --v51_dae_hidden 64 \
    --v51_dae_num_layers 2 \
    --v51_dae_dropout 0.1 \
    --v51_dae_n_experts 2 \
    --v51_dae_min_weight 0.05 \
    --v51_dae_loss_weight 0.005 \
    --no_v51_dae_identity_bypass

# 4. Smaller hidden / shallower gate.
run_variant "small_gate" \
    --v51_dae_hidden 32 \
    --v51_dae_num_layers 1 \
    --v51_dae_dropout 0.1 \
    --v51_dae_n_experts 2 \
    --v51_dae_identity_bypass \
    --v51_dae_min_weight 0.05 \
    --v51_dae_loss_weight 0.005

echo "$(date -Iseconds) v51 DAE ablation matrix complete"
