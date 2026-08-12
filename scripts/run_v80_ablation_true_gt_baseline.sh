#!/usr/bin/env bash
# Wait until the v25 true-GT ablation 1 (baseline fix) has finished and the
# local RTX 4090 is idle, then run the same winning recipe on the v80
# architecture.
#
# This wrapper enforces the project rule of at most ONE training task on the
# local GPU at a time. It is prepared but NOT executed here.
#
# Usage (manual, once v25 ablation 1 has finished):
#   bash scripts/run_v80_ablation_true_gt_baseline.sh
#
# To run detached:
#   nohup bash scripts/run_v80_ablation_true_gt_baseline.sh \
#       > outputs/ablations/v80_true_gt_baseline_fix_nohup.log 2>&1 &
set -euo pipefail

POLL_SEC=${POLL_SEC:-60}
LOG_DIR="outputs/ablations"
mkdir -p "$LOG_DIR"

ABLA_LOG="${LOG_DIR}/v80_true_gt_baseline_fix_wrapper_$(date +%Y%m%d_%H%M%S).log"
ABLA_MARKER="v25_true_gt_baseline_fix"
ABLA_SCRIPT="scripts/run_v25_ablation_true_gt_baseline.sh"

PYTHON=${PYTHON:-python}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ABLA_LOG"
}

# Return 0 if the v25 ablation 1 process is still running.
is_ablation1_running() {
    ps -ef 2>/dev/null \
        | grep -v grep \
        | grep -E "${ABLA_SCRIPT}|${ABLA_MARKER}" \
        >/dev/null 2>&1
}

# Return 0 if there are no python compute processes on the GPU.
is_gpu_idle() {
    local count
    count=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -i "python" \
        | wc -l)
    [[ "$count" -eq 0 ]]
}

log "Waiting for v25 ablation 1 (${ABLA_MARKER}) to finish and the GPU to become idle..."
while true; do
    abla_busy=0
    gpu_busy=0

    if is_ablation1_running; then
        abla_busy=1
    fi

    if ! is_gpu_idle; then
        gpu_busy=1
    fi

    if [[ "$abla_busy" -eq 0 && "$gpu_busy" -eq 0 ]]; then
        log "v25 ablation 1 finished and GPU is idle."
        break
    fi

    [[ "$abla_busy" -ne 0 ]] && log "v25 ablation 1 still running."
    [[ "$gpu_busy" -ne 0 ]] && log "GPU not idle (python compute process still present)."

    log "Polling again in ${POLL_SEC}s..."
    sleep "$POLL_SEC"
done

log "Starting v80 true-GT baseline ablation."

# v80 true-GT baseline ablation using the winning v25 recipe:
#   - train_samples 4096, weight_decay 1e-4, lr 5e-4, 2-epoch warmup
#   - early stopping (patience 3)
#   - reduced/milder outlier augmentation (prob 0.15, max_views 1)
#   - v25 geometry loss weight reduced to 0.05 + v25 dropout 0.2
#   - full DLT reweight stack from the v25 recipe
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
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
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_dropout 0.1 \
    --v50_sefh_identity_init_gate \
    --v50_sefh_loss_weight 0.0 --v50_sefh_aleatoric_weight 0.0 \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 --v51_cdsvr_num_heads 4 --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_use_domain_label --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 --v52_uwt_n_layers 2 --v52_uwt_weight_type per_view_joint \
    --v52_uwt_use_geometry_bias --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init --v52_uwt_min_weight 0.05 --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 0.0001 \
    --use_v80_view_reliability \
    --v80_vrbt_hidden 64 --v80_vrbt_n_layers 2 \
    --v80_vrbt_weight_type per_view_joint \
    --v80_vrbt_use_geometry_bias --v80_vrbt_use_feature_bias \
    --v80_vrbt_identity_init --v80_vrbt_min_weight 0.05 \
    --num_workers 0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 5e-4 --lr_cosine --lr_warmup_epochs 2 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v80_true_gt_baseline_fix.pth \
    > outputs/ablations/v80_true_gt_baseline_fix.log 2>&1

log "v80 true-GT baseline ablation finished."
