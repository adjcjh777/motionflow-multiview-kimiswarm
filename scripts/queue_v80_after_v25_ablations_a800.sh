#!/usr/bin/env bash
# A800-D queue: wait for the two running v25 true-GT ablations to finish,
# then launch the v80 stronger-regularisation ablation on the first free GPU.
#
# This script is designed to be started from the local WSL repo (or anywhere
# with SSH access to a800-D). It does not touch the A800 projects/ or Docker
# state until it actually launches the v80 ablation.
#
# Usage
# -----
#   # Start the watcher (foreground)
#   bash scripts/queue_v80_after_v25_ablations_a800.sh
#
#   # Start the watcher detached
#   nohup bash scripts/queue_v80_after_v25_ablations_a800.sh \
#       > outputs/queue_v80_after_v25_ablations_a800.log 2>&1 &
#
#   # Use a custom poll interval (seconds)
#   POLL_SEC=120 bash scripts/queue_v80_after_v25_ablations_a800.sh
#
# Behaviour
# ---------
#   1. Polls a800-D every ${POLL_SEC}s.
#   2. Waits until no python training process named like the v25 ablations is
#      running and both expected output checkpoints exist.
#   3. Picks the first GPU that is free (no training process, < 1 GiB used).
#   4. Launches scripts/run_v80_ablation_true_gt_regularization_a800.sh on that
#      GPU via SSH in a tmux session named v80_true_gt_regularization_gpu<N>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

A800_HOST="${A800_HOST:-a800-D}"
A800_REPO="${A800_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}"
POLL_SEC="${POLL_SEC:-60}"
FREE_MEMORY_MB="${FREE_MEMORY_MB:-1000}"
V80_SCRIPT="scripts/run_v80_ablation_true_gt_regularization_a800.sh"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

a800_ssh() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$A800_HOST" "$1"
}

# Return 0 if a python process whose command line contains $name is running on a800-D.
a800_is_running() {
    local name="$1"
    local count
    count=$(a800_ssh "ps -ef | grep -v grep | grep 'python.*${name}' | wc -l" || echo 0)
    [[ "$count" -gt 0 ]]
}

# Return the list of free GPU indices on a800-D (no training process, < 1 GiB used).
a800_free_gpus() {
    a800_ssh "
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | grep -i python > /tmp/a800_compute_apps_\$$.csv 2>/dev/null || true
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null
    " | awk -v mem_threshold="$FREE_MEMORY_MB" 'NF==2 { gsub(/ MiB/,"",$2); if ($2 < mem_threshold) print $1 }'
}

log "A800-D v80 queue started."
log "Waiting for v25 ablations to finish on ${A800_HOST}..."
log "Poll interval: ${POLL_SEC}s"

while true; do
    baseline_running=false
    geom_running=false

    if a800_is_running "v25_true_gt_baseline_fix"; then
        baseline_running=true
    fi

    if a800_is_running "v25_true_gt_geometry_regularization_a800"; then
        geom_running=true
    fi

    if ! $baseline_running && ! $geom_running; then
        log "Both v25 ablations appear to have finished."
        break
    fi

    if $baseline_running; then
        log "v25_true_gt_baseline_fix still running."
    fi
    if $geom_running; then
        log "v25_true_gt_geometry_regularization_a800 still running."
    fi

    log "Polling again in ${POLL_SEC}s..."
    sleep "$POLL_SEC"
done

# Extra safety: wait one more poll interval for the checkpoints to be written.
log "Waiting an additional ${POLL_SEC}s for checkpoints to flush..."
sleep "$POLL_SEC"

# Pick a free GPU.
free_gpu=$(a800_free_gpus | head -n 1 || true)
if [[ -z "$free_gpu" ]]; then
    log "ERROR: no free GPU on ${A800_HOST} after v25 ablations finished."
    exit 1
fi

log "GPU ${free_gpu} is free. Launching v80 regularisation ablation."

# Launch in a tmux session so it survives SSH logout.
session_name="v80_true_gt_regularization_gpu${free_gpu}"
a800_ssh "
    cd ${A800_REPO} && \
    tmux new-session -d -s ${session_name} \
        'CUDA_VISIBLE_DEVICES=${free_gpu} bash ${V80_SCRIPT}'
"

log "Launched ${V80_SCRIPT} on ${A800_HOST} GPU ${free_gpu} in tmux session ${session_name}."
log "Attach with: ssh ${A800_HOST} -t 'tmux attach -t ${session_name}'"
