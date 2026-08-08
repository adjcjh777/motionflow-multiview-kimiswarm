#!/usr/bin/env bash
# Launch v29 A800 ablation matrix (v29a, v29b, v29d) on free GPUs.
# GPU0 is reserved for the full v29 SEH-MV run; GPUs 1-3 are used here.
set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
BASE="$REPO/experiments/train_omniview_fusion_v5_webbridge_multi.py"

# All flags on one line so the tmux command string is parsed correctly.
COMMON_FLAGS="--use_mixed_loader --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml --use_full_precision_dlt --use_robust_dlt_reweight --use_irls_reweight --use_domain_embedding --use_deformable_cross_view_attention_v18 --use_multiview_geometry_fusion_v25 --v25_dropout 0.2 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment --num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true --use_entropy_regularization true --attention_entropy_weight 0.01 --use_camera_view_embedding --use_set_view_aggregator --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 --outlier_view_prob 0.3 --outlier_view_max_views 1 --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0"

launch_if_free() {
    local session="$1"
    local gpu="$2"
    local extra="$3"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session $session already exists; skipping."
        return
    fi
    tmux new-session -d -s "$session" -n v29 "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu python3 -u $BASE $COMMON_FLAGS $extra"
    echo "Launched $session on GPU $gpu."
}

# v29a: hierarchical encoder only (no TTE, no physical loss).
launch_if_free v29a_gpu1 1 "--use_hierarchical_multiview_v29 --output outputs/omniview_fusion_v29a_hierarchical_only_a800.pth > outputs/omniview_fusion_v29a_hierarchical_only_a800.log 2>&1"

# v29b: hierarchical encoder + TTE.
launch_if_free v29b_gpu2 2 "--use_hierarchical_multiview_v29 --use_test_time_self_evolution_v29 --output outputs/omniview_fusion_v29b_hierarchical_tte_a800.pth > outputs/omniview_fusion_v29b_hierarchical_tte_a800.log 2>&1"

# v29d: TTE + physical temporal loss, no hierarchical encoder.
launch_if_free v29d_gpu3 3 "--use_test_time_self_evolution_v29 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 --output outputs/omniview_fusion_v29d_tte_physical_only_a800.pth > outputs/omniview_fusion_v29d_tte_physical_only_a800.log 2>&1"

# GPU0 reserved for the full v29 SEH-MV run (use launch_v29_a800_tmux.sh).
