#!/usr/bin/env bash
# A800-side watcher: wait for v25 true-GT v2 medium training to finish, then
# launch v86 sparse cross-domain v2 medium on the first free project GPU.
#
# Usage (on A800):
#   nohup bash scripts/launch_v86_after_v25_a800.sh > outputs/launch_v86_after_v25_a800.log 2>&1 &

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
V25_LOG="outputs/ablations/v25_true_gt_v2_medium_a800.log"
V25_CKPT="outputs/ablations/v25_true_gt_v2_medium_a800.pth"
V25_SESSION="v25_true_gt_v2_medium_a800"
V86_SCRIPT="scripts/run_v86_sparse_cross_domain_v2_medium_a800.sh"
V86_LOG="outputs/ablations/v86_sparse_cross_domain_v2_medium_a800.log"

POLL_SEC=60

# GPU selection: prefer 6, then 7.
select_free_gpu() {
    for idx in 6 7; do
        local used
        used=$(nvidia-smi --id="${idx}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
        if [[ "${used}" -lt 1000 ]]; then
            echo "${idx}"
            return 0
        fi
    done
    return 1
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "${REPO}"

log "A800 watcher started. Waiting for v25 training to finish."
log "Monitoring tmux session '${V25_SESSION}' and log ${V25_LOG}."

while true; do
    # v25 is considered done when the tmux session no longer exists AND the
    # checkpoint has been written (or the log contains 'Early stopping'/'best val').
    session_exists="$(tmux has-session -t "${V25_SESSION}" 2>/dev/null && echo yes || echo no)"
    log_finished="$(grep -qE 'Early stopping|Early-stopped|best val|Training complete' "${V25_LOG}" 2>/dev/null && echo yes || echo no)"
    ckpt_exists="$(test -f "${V25_CKPT}" && echo yes || echo no)"

    if [[ "${session_exists}" == "no" ]] && { [[ "${log_finished}" == "yes" ]] || [[ "${ckpt_exists}" == "yes" ]]; }; then
        log "v25 appears to have finished (session gone, log_finished=${log_finished}, ckpt_exists=${ckpt_exists})."

        # Wait for a project GPU to become free.
        FREE_GPU=""
        while true; do
            if FREE_GPU="$(select_free_gpu)"; then
                break
            fi
            log "No free GPU 6/7; waiting ${POLL_SEC}s..."
            sleep "${POLL_SEC}"
        done

        log "GPU ${FREE_GPU} is free. Launching v86 training."
        V86_SESSION="v86_sparse_cross_domain_v2_gpu${FREE_GPU}"
        tmux has-session -t "${V86_SESSION}" 2>/dev/null || tmux new-session -d -s "${V86_SESSION}" \
            "cd ${REPO} && CUDA_VISIBLE_DEVICES=${FREE_GPU} bash ${V86_SCRIPT} > ${V86_LOG} 2>&1"

        log "v86 launched in tmux session ${V86_SESSION}. Exiting."
        exit 0
    fi

    log "v25 still running (session=${session_exists}, log_finished=${log_finished}, ckpt_exists=${ckpt_exists}); polling in ${POLL_SEC}s..."
    sleep "${POLL_SEC}"
done
