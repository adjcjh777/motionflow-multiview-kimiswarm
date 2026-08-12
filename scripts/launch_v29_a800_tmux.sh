#!/usr/bin/env bash
# Launch v29 full-scale A800-D run in a tmux session.
# This script is intended to be run FROM the A800-D host (or via SSH).
set -euo pipefail

SESSION="v29_gpu5"
REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
LOG="outputs/omniview_fusion_v29_full_seh_mv_a800.log"
OUTPUT="outputs/omniview_fusion_v29_full_seh_mv_a800.pth"

# Create tmux session if it does not already exist.
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session $SESSION already exists. Attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s $SESSION -n v29 "cd $REPO && CUDA_VISIBLE_DEVICES=5 python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/deprecated/circular/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_hierarchical_multiview_v29 \
    --use_test_time_self_evolution_v29 --v27_tte_n_iters 3 \
    --use_physical_space_temporal_loss_v29 \
    --num_workers 4 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 24 --train_samples 4000 --val_stride 10 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 --variable_view_max_views 14 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 3 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output $OUTPUT \
    > $LOG 2>&1"

echo "Launched tmux session $SESSION on A800-D."
echo "Attach: tmux attach -t $SESSION"
echo "Monitor: tail -f $REPO/$LOG"
