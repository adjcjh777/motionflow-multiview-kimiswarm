#!/usr/bin/env bash
# Launch v86 sparse-view / cross-domain medium run on the corrected H36M true-GT
# v2 protocol once GPU 6 or 7 is free on A800-D.
#
# Usage
# -----
#   # Poll until GPU 6/7 is free, then launch in a tmux session on A800-D.
#   bash scripts/launch_v86_sparse_cross_domain_v2_medium_a800.sh
#
#   # Detached launch from local/WSL
#   nohup bash scripts/launch_v86_sparse_cross_domain_v2_medium_a800.sh \
#       > outputs/launch_v86_sparse_cross_domain_v2_medium_a800.log 2>&1 &
#
# The script never uses GPUs 0-5 (project policy).

set -euo pipefail

SSH_HOST="a800-D"
REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
# The venv lives in the base repo, not the -iter20 worktree.
PYTHON="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"

# GPU availability threshold.  A GPU is considered free when no process is
# running on it and its memory is essentially idle.
MIN_FREE_MIB=60000
POLL_INTERVAL_SEC=60
TIMEOUT_SEC=${TIMEOUT_SEC:-0}

TARGET_GPUS="6 7"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

Wait for GPU 6/7 availability on ${SSH_HOST}, then launch the v86 sparse-view /
cross-domain medium run on the H36M true-GT v2 protocol.

Options:
  -t, --timeout SECONDS   Exit with error after SECONDS (default 0 = no timeout).
  -i, --interval SECONDS  Polling interval in seconds (default ${POLL_INTERVAL_SEC}).
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--timeout)
            TIMEOUT_SEC="$2"
            shift 2
            ;;
        -i|--interval)
            POLL_INTERVAL_SEC="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log "ERROR: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

START_TIME=${SECONDS}

is_elapsed() {
    if [[ "${TIMEOUT_SEC}" -gt 0 && $((SECONDS - START_TIME)) -ge "${TIMEOUT_SEC}" ]]; then
        return 0
    fi
    return 1
}

a800_ssh() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_HOST}" "$@"
}

# Verify the remote host is reachable.
if ! a800_ssh "true" </dev/null >/dev/null 2>&1; then
    log "ERROR: Cannot reach ${SSH_HOST} via SSH." >&2
    exit 1
fi

log "=== v86 sparse cross-domain v2 medium launch ==="
log "Polling ${SSH_HOST} for a free GPU among {${TARGET_GPUS// /,}}..."

FREE_GPU=""
while true; do
    if is_elapsed; then
        log "ERROR: Timeout reached after ${TIMEOUT_SEC}s without finding a free GPU among {${TARGET_GPUS// /,}}." >&2
        exit 1
    fi

    GPU_INFO=$(a800_ssh "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits" </dev/null 2>/dev/null) || {
        log "ERROR: Failed to query GPU status on ${SSH_HOST}." >&2
        exit 1
    }

    while IFS=',' read -r idx free_mib util_pct; do
        idx=$(echo "${idx}" | xargs)
        free_mib=$(echo "${free_mib}" | xargs)
        util_pct=$(echo "${util_pct}" | xargs)

        # Restrict to project GPUs only.
        if [[ ! " ${TARGET_GPUS} " =~ \ ${idx}\  ]]; then
            continue
        fi

        # util_pct may be "0" or "0 %"; strip % just in case.
        util_num="${util_pct%%%}"

        if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" && "${util_num}" == "0" ]]; then
            # Double-check there are no compute processes on this GPU.
            PROC_COUNT=$(a800_ssh "nvidia-smi --id=${idx} --query-compute-apps=pid --format=csv,noheader | wc -l" </dev/null 2>/dev/null) || PROC_COUNT=999
            if [[ "${PROC_COUNT}" -eq 0 ]]; then
                FREE_GPU="${idx}"
                break
            fi
        fi
    done <<< "${GPU_INFO}"

    if [[ -n "${FREE_GPU}" ]]; then
        log "GPU ${FREE_GPU} is free (memory >= ${MIN_FREE_MIB} MiB, util 0%, no processes)."
        break
    fi

    log "No free GPU; sleeping ${POLL_INTERVAL_SEC}s..."
    sleep "${POLL_INTERVAL_SEC}"
done

SESSION="v86_sparse_cross_domain_v2_gpu${FREE_GPU}"

log "Launching v86 sparse cross-domain v2 medium on GPU ${FREE_GPU} (tmux session: ${SESSION})."

# Build the run script on the remote side so the command is easy to review and
# the log path is deterministic.
LAUNCH_SCRIPT="/tmp/run_v86_sparse_cross_domain_v2_medium_gpu${FREE_GPU}.sh"

a800_ssh "cat > ${LAUNCH_SCRIPT} <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd ${REPO}
mkdir -p outputs/ablations

CUDA_VISIBLE_DEVICES=${FREE_GPU} /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \\
    --use_mixed_loader \\
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \\
    --num_domains 1 \\
    --use_full_precision_dlt \\
    --use_robust_dlt_reweight \\
    --use_irls_reweight \\
    --use_domain_embedding \\
    --use_deformable_cross_view_attention_v18 \\
    --use_multiview_geometry_fusion_v25 \\
    --v25_geom_loss_weight 0.05 \\
    --v25_dropout 0.2 \\
    --v25_use_geometry_attention \\
    --v25_use_learned_depth_triangulation \\
    --v25_use_geometry_bundle_adjustment \\
    --use_random_view_dropout_v85 \\
    --v85_dropout_prob 0.3 \\
    --v85_min_views 2 \\
    --v85_use_count_embedding \\
    --use_v86_strong_count_conditioning \\
    --v86_count_hidden 64 \\
    --v86_count_n_layers 2 \\
    --v86_count_dropout 0.1 \\
    --use_v86_separate_sparse_view_head \\
    --v86_ssv_head_hidden 128 \\
    --v86_ssv_head_n_layers 2 \\
    --v86_ssv_head_dropout 0.1 \\
    --v86_ssv_head_use_count_embedding \\
    --num_workers 4 \\
    --d 128 \\
    --residual_hidden 256 \\
    --n_st_layers 3 \\
    --graph_num_layers 1 \\
    --n_joint_layers 1 \\
    --n_heads 4 \\
    --clip_len 13 \\
    --epochs 20 \\
    --batch_size 16 \\
    --train_samples 4096 \\
    --val_stride 20 \\
    --lr 1e-4 \\
    --lr_cosine \\
    --lr_warmup_epochs 4 \\
    --lr_min 1e-6 \\
    --max_grad_norm 1.0 \\
    --ema_decay 0.999 \\
    --weight_decay 1e-4 \\
    --early_stopping_patience 3 \\
    --early_stopping_min_delta 0.001 \\
    --use_multiscale_fusion true \\
    --use_camera_conditioning true \\
    --use_epipolar_bias true \\
    --use_context_visibility true \\
    --use_skeleton_residual true \\
    --use_rotation_correction true \\
    --use_entropy_regularization true \\
    --attention_entropy_weight 0.01 \\
    --use_camera_view_embedding \\
    --use_set_view_aggregator \\
    --use_variable_view_training \\
    --variable_view_min_views 2 \\
    --variable_view_max_views 4 \\
    --variable_view_max_views_start 4 \\
    --variable_view_curriculum_alpha 2.0 \\
    --pa_loss_weight 0.5 \\
    --monotonic_loss_weight 0.1 \\
    --monotonic_margin 5.0 \\
    --reproj_loss_weight 0.1 \\
    --reproj_warmup_epochs 1 \\
    --aleatoric_reproj_loss_weight 0.1 \\
    --outlier_view_prob 0.15 \\
    --outlier_view_max_views 1 \\
    --outlier_view_offset_std 10.0 \\
    --outlier_view_noise_std 15.0 \\
    --output outputs/ablations/v86_sparse_cross_domain_v2_medium_a800.pth \\
    > outputs/ablations/v86_sparse_cross_domain_v2_medium_a800.log 2>&1
EOF
chmod +x ${LAUNCH_SCRIPT}

tmux has-session -t ${SESSION} 2>/dev/null || tmux new-session -d -s ${SESSION} -n v86 "bash ${LAUNCH_SCRIPT}"
"

log "Launched v86 sparse cross-domain v2 medium on GPU ${FREE_GPU}."
log "Attach with:  ssh ${SSH_HOST} \"tmux attach -t ${SESSION}\""
log "Log file:     ${REPO}/outputs/ablations/v86_sparse_cross_domain_v2_medium_a800.log"
