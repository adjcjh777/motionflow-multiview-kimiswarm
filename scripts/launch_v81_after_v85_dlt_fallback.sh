#!/usr/bin/env bash
# Watcher: after the v85 DLT-fallback variable-view eval finishes,
# launch v81 true-GT v2 medium training on the first free project GPU.
set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
TARGET_SCRIPT="scripts/run_v81_true_gt_v2_medium_a800.sh"
TMUX_NAME="v81_true_gt_v2_medium_a800"
POLL_SEC=60

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

find_free_gpu() {
    for gpu in 6 7; do
        local util mem procs
        read -r util mem <<< "$(nvidia-smi --id=${gpu} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')"
        procs="$(nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
        util="${util//%/}"; util="${util:-100}"
        mem="${mem//MiB/}"; mem="${mem:-999999}"
        procs="${procs// /}"
        if [[ "${util}" =~ ^[0-9]+$ && "${mem}" =~ ^[0-9]+$ && "${procs}" =~ ^[0-9]+$ ]] \
           && [[ "${util}" -lt 10 ]] \
           && [[ "${mem}" -lt 1000 ]] \
           && [[ "${procs}" -eq 0 ]]; then
            echo "${gpu}"
            return 0
        fi
    done
    return 1
}

cd "${REPO}"
log "Watcher started. Waiting for v85 DLT-fallback to finish and a free GPU 6/7."

while true; do
    if pgrep -f "experiments/eval_variable_views.py.*v85_random_view_dropout_medium_a800" >/dev/null 2>&1; then
        log "v85 DLT-fallback still running; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
        continue
    fi

    free_gpu="$(find_free_gpu)"
    if [[ -z "${free_gpu}" ]]; then
        log "v85 DLT-fallback finished but no free GPU 6/7; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
        continue
    fi

    log "GPU ${free_gpu} is free. Launching v81 true-GT v2 medium."

    tmux kill-session -t "${TMUX_NAME}" 2>/dev/null || true
    tmux new-session -d -s "${TMUX_NAME}" "cd ${REPO} && CUDA_VISIBLE_DEVICES=${free_gpu} bash ${TARGET_SCRIPT}"

    log "Launched v81 on GPU ${free_gpu} in tmux session ${TMUX_NAME}."
    exit 0
done
