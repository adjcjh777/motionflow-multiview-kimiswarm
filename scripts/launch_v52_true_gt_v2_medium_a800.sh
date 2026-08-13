#!/usr/bin/env bash
# Launch wrapper for the v52 true-GT v2 medium A800 run.
#
# Waits until GPU 6 or GPU 7 is free (project GPUs only; never touches 0-5),
# then launches scripts/run_v52_true_gt_v2_medium_a800.sh on the first
# available GPU in a persistent tmux session.
#
# Usage
# -----
#   # Foreground (blocks until a GPU is free and the job is launched)
#   bash scripts/launch_v52_true_gt_v2_medium_a800.sh
#
#   # Detached watcher
#   nohup bash scripts/launch_v52_true_gt_v2_medium_a800.sh \
#       > outputs/launch_v52_true_gt_v2_medium_a800.log 2>&1 &
#
# Environment overrides
# ---------------------
#   A800_HOST           SSH host alias for A800-D (default: a800-D)
#   A800_REPO           Absolute path to the A800 repo checkout
#                       (default: /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20)
#   POLL_SEC            Polling interval in seconds (default: 60)
#   FREE_MEMORY_MB      Memory threshold below which a GPU is considered free
#                       (default: 1000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

A800_HOST="${A800_HOST:-a800-D}"
A800_REPO="${A800_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}"
POLL_SEC="${POLL_SEC:-60}"
FREE_MEMORY_MB="${FREE_MEMORY_MB:-1000}"
UTIL_THRESHOLD=10

# Project GPU policy: only 6 and 7 may be used.
ALLOWED_GPUS=(6 7)

TARGET_SCRIPT="scripts/run_v52_true_gt_v2_medium_a800.sh"
TMUX_NAME="v52_true_gt_v2_medium_a800"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

a800_ssh() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "${A800_HOST}" "$1"
}

# Return the first GPU index (6 or 7) whose utilization is below the threshold,
# whose memory usage is below FREE_MEMORY_MB, and which has no compute
# processes.  Empty string if none free.
find_free_gpu() {
    local gpu util mem proc_count
    for gpu in "${ALLOWED_GPUS[@]}"; do
        read -r util mem <<< "$(a800_ssh "nvidia-smi --id=${gpu} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null" | tr -d ' ' | tr ',' ' ')"

        # Strip units / handle empty values.
        util="${util//%/}"
        mem="${mem//MiB/}"
        util="${util:-100}"
        mem="${mem:-999999}"

        # Also verify no compute processes are running on this GPU.
        proc_count="$(a800_ssh "nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l")"
        proc_count="${proc_count// /}"

        if [[ -n "${util}" && "${util}" =~ ^[0-9]+$ ]] \
           && [[ -n "${mem}" && "${mem}" =~ ^[0-9]+$ ]] \
           && [[ "${util}" -lt "${UTIL_THRESHOLD}" ]] \
           && [[ "${mem}" -lt "${FREE_MEMORY_MB}" ]] \
           && [[ "${proc_count}" -eq 0 ]]; then
            echo "${gpu}"
            return 0
        fi
    done
    echo ""
}

log "Launch watcher started for v52 true-GT v2 medium run on ${A800_HOST}."
log "Allowed GPUs: ${ALLOWED_GPUS[*]}"
log "Polling every ${POLL_SEC}s until a project GPU is free."

while true; do
    free_gpu="$(find_free_gpu)"

    if [[ -n "${free_gpu}" ]]; then
        log "GPU ${free_gpu} is free. Launching ${TARGET_SCRIPT} with CUDA_VISIBLE_DEVICES=${free_gpu}."

        a800_ssh "
            cd ${A800_REPO} && \
            tmux kill-session -t ${TMUX_NAME} 2>/dev/null || true && \
            tmux new-session -d -s ${TMUX_NAME} \
                'CUDA_VISIBLE_DEVICES=${free_gpu} bash ${TARGET_SCRIPT}'
        "

        log "Launched ${TARGET_SCRIPT} on ${A800_HOST} GPU ${free_gpu} in tmux session '${TMUX_NAME}'."
        log "Attach with: ssh ${A800_HOST} -t 'tmux attach -t ${TMUX_NAME}'"
        exit 0
    fi

    log "GPUs 6/7 are both busy; polling again in ${POLL_SEC}s..."
    sleep "${POLL_SEC}"
done
