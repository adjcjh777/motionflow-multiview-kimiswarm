#!/usr/bin/env bash
# Queue v52 true-GT H36M A800 run to start after v46 on GPU 4 finishes.
#
# Usage
# -----
#   # Run locally on A800-D; starts immediately and polls until v46 is done.
#   nohup bash scripts/queue_v52_after_v46.sh >> outputs/ablations/queue_v52_after_v46.log 2>&1 &
#
# The wrapper waits until no process is writing to
# outputs/ablations/v46_true_gt_h36m_a800.pth, then launches
# scripts/run_v52_true_gt_h36m_a800.sh on CUDA_VISIBLE_DEVICES=4.

set -euo pipefail

REPO_ROOT=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
cd "$REPO_ROOT"

mkdir -p outputs/ablations

V46_OUTPUT="outputs/ablations/v46_true_gt_h36m_a800.pth"
V46_LOG="outputs/ablations/v46_true_gt_h36m_a800.log"
QUEUE_LOG="outputs/ablations/queue_v52_after_v46.log"

POLL_INTERVAL_SEC=60

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

is_v46_running() {
    # Match the training process for v46 by its unique output checkpoint path.
    pgrep -f "$V46_OUTPUT" >/dev/null 2>&1 || return 1
    return 0
}

is_v52_running() {
    pgrep -f "v52_true_gt_h36m_a800.pth" >/dev/null 2>&1 || return 1
    return 0
}

log "Queue wrapper started for v52 after v46."
log "Waiting for v46 process to finish (polling every ${POLL_INTERVAL_SEC}s)..."

while is_v46_running; do
    sleep "$POLL_INTERVAL_SEC"
done

log "v46 process no longer detected."

if is_v52_running; then
    log "ERROR: v52 is already running; aborting duplicate launch."
    exit 1
fi

if [ -f "$V46_OUTPUT" ]; then
    V46_SIZE=$(stat -c%s "$V46_OUTPUT" 2>/dev/null || echo "unknown")
    log "v46 checkpoint present: $V46_OUTPUT (size: $V46_SIZE bytes)"
fi

log "Launching v52 on GPU 4."
CUDA_VISIBLE_DEVICES=4 nohup bash scripts/run_v52_true_gt_h36m_a800.sh > outputs/ablations/v52_true_gt_h36m_a800_queued.log 2>&1 &
V52_PID=$!
log "v52 launched with PID $V52_PID; log: outputs/ablations/v52_true_gt_h36m_a800_queued.log"
