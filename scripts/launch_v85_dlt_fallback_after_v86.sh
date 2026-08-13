#!/usr/bin/env bash
# A800-local watcher: after the v86 no-count-embedding medium training finishes,
# run the v85 DLT-fallback variable-view evaluation on the first free project GPU.
#
# Usage (on A800):
#   nohup bash scripts/launch_v85_dlt_fallback_after_v86.sh > outputs/launch_v85_dlt_fallback_after_v86.log 2>&1 &

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
V86_LOG="outputs/ablations/v86_no_count_embedding_medium_a800.log"
V86_CKPT="outputs/ablations/v86_no_count_embedding_medium_a800.pth"
V85_CKPT="outputs/ablations/v85_random_view_dropout_medium_a800_final.pth"
V85_EVAL_SCRIPT="scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh"

POLL_SEC=60

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Prefer GPU 6, fallback to 7.
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

cd "${REPO}"

log "A800 watcher started. Waiting for v86 training to finish."
log "Monitoring ${V86_LOG} and checkpoint ${V86_CKPT}."

while true; do
    log_finished="$(grep -qE 'Early stopping|Early-stopped|best val|Training complete' "${V86_LOG}" 2>/dev/null && echo yes || echo no)"
    ckpt_exists="$(test -f "${V86_CKPT}" && echo yes || echo no)"

    if [[ "${log_finished}" == "yes" ]] || [[ "${ckpt_exists}" == "yes" ]]; then
        log "v86 appears to have finished (log_finished=${log_finished}, ckpt_exists=${ckpt_exists})."

        # Wait for a project GPU to become free.
        FREE_GPU=""
        while true; do
            if FREE_GPU="$(select_free_gpu)"; then
                break
            fi
            log "No free GPU 6/7; waiting ${POLL_SEC}s..."
            sleep "${POLL_SEC}"
        done

        log "GPU ${FREE_GPU} is free. Launching v85 DLT-fallback variable-view eval."

        # Ensure v85 checkpoint symlink/path exists.
        if [[ ! -f "outputs/ablations/v85_random_view_dropout_medium_a800.pth" ]] && [[ -f "${V85_CKPT}" ]]; then
            ln -sf "v85_random_view_dropout_medium_a800_final.pth" "outputs/ablations/v85_random_view_dropout_medium_a800.pth"
            log "Created symlink for v85 checkpoint."
        fi

        # Modify eval script to use the free GPU.
        sed -i "s/CUDA_VISIBLE_DEVICES=.*/CUDA_VISIBLE_DEVICES=${FREE_GPU}/" "${V85_EVAL_SCRIPT}"
        bash "${V85_EVAL_SCRIPT}"

        log "v85 DLT-fallback eval launched on GPU ${FREE_GPU}. Exiting."
        exit 0
    fi

    log "v86 still running (log_finished=${log_finished}, ckpt_exists=${ckpt_exists}); polling in ${POLL_SEC}s..."
    sleep "${POLL_SEC}"
done
