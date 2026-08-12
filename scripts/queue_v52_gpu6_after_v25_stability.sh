#!/usr/bin/env bash
# Queue v52 true-GT H36M A800 run to start on GPU 6 once it is free.
#
# Usage
# -----
#   nohup bash scripts/queue_v52_gpu6_after_v25_stability.sh >> outputs/ablations/queue_v52_gpu6_nohup.log 2>&1 &
#
# Waits until no MotionFlow training process is bound to GPU 6, then launches
# scripts/run_v52_true_gt_h36m_a800.sh with CUDA_VISIBLE_DEVICES=6.

set -euo pipefail

REPO_ROOT=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
cd "$REPO_ROOT"

mkdir -p outputs/ablations

LOG="outputs/ablations/queue_v52_gpu6.log"
POLL_INTERVAL_SEC=60

# Patterns used to detect occupying jobs.
GPU6_BUSY_PATTERN="v25_true_gt_stability_a800\.pth"
V52_PATTERN="v52_true_gt_h36m_a800\.pth"

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

is_gpu6_busy() {
    # GPU 6 is busy if the v25 stability parent or worker process is still alive.
    pgrep -f "$GPU6_BUSY_PATTERN" >/dev/null 2>&1
}

is_v52_running() {
    pgrep -f "$V52_PATTERN" >/dev/null 2>&1
}

log "Queue wrapper started for v52 on GPU 6."

if is_v52_running; then
    log "ERROR: v52 is already running elsewhere; aborting duplicate GPU 6 launch."
    exit 1
fi

log "Waiting for GPU 6 to become free (polling every ${POLL_INTERVAL_SEC}s)..."

while is_gpu6_busy; do
    log "GPU 6 still occupied by v25 stability; sleeping ${POLL_INTERVAL_SEC}s."
    sleep "$POLL_INTERVAL_SEC"
done

if is_v52_running; then
    log "ERROR: v52 launched by another process while waiting; aborting duplicate GPU 6 launch."
    exit 1
fi

log "GPU 6 is free. Launching v52 on GPU 6."

CUDA_VISIBLE_DEVICES=6 nohup bash scripts/run_v52_true_gt_h36m_a800.sh > outputs/ablations/v52_true_gt_h36m_a800_gpu6_nohup.log 2>&1 &
V52_PID=$!
log "v52 launched on GPU 6 with PID ${V52_PID}; nohup log: outputs/ablations/v52_true_gt_h36m_a800_gpu6_nohup.log"
log "Training log (script internal): outputs/ablations/v52_true_gt_h36m_a800.log"
