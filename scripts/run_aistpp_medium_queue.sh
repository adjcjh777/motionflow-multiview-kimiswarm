#!/usr/bin/env bash
# Sequential AIST++ medium queue: v80 first, then v25.
#
# Usage (once the local RTX 4090 GPU is free and no other training is running):
#   bash scripts/run_aistpp_medium_queue.sh
#
# To run detached:
#   nohup bash scripts/run_aistpp_medium_queue.sh > outputs/run_aistpp_medium_queue_nohup.log 2>&1 &
#
# This script enforces the project rule of at most ONE training task on the
# local GPU at a time.  It polls the GPU, launches v80 AIST++ medium, waits for
# the GPU to become idle again, then launches v25 AIST++ medium.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_GPU="${SCRIPT_DIR}/sota_baselines/check_gpu_free.sh"

LOG_DIR="outputs/run_aistpp_medium_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

wait_for_gpu() {
    if [[ ! -f "$CHECK_GPU" ]]; then
        log "ERROR: GPU check helper missing: $CHECK_GPU"
        exit 1
    fi

    while true; do
        if bash "$CHECK_GPU" >"${LOG_DIR}/gpu_check.log" 2>&1; then
            log "GPU is idle."
            return 0
        fi
        log "GPU is busy; waiting... (see ${LOG_DIR}/gpu_check.log)"
        sleep 60
    done
}

log "=== AIST++ medium queue started ==="
log "Logs directory: $LOG_DIR"

# v80 AIST++ medium
log "=== Queueing v80 AIST++ medium ==="
wait_for_gpu
log "Starting v80 AIST++ medium"
bash "${SCRIPT_DIR}/run_v80_aistpp_train_val_local_4090.sh" >"$LOG_DIR/v80.log" 2>&1
V80_EXIT=$?
log "v80 AIST++ medium finished (exit code: $V80_EXIT)"

# v25 AIST++ medium
log "=== Queueing v25 AIST++ medium ==="
wait_for_gpu
log "Starting v25 AIST++ medium"
bash "${SCRIPT_DIR}/run_v25_aistpp_train_val_local_4090.sh" >"$LOG_DIR/v25.log" 2>&1
V25_EXIT=$?
log "v25 AIST++ medium finished (exit code: $V25_EXIT)"

# Final summary
FINAL_STATUS="success"
if [[ "$V80_EXIT" -ne 0 || "$V25_EXIT" -ne 0 ]]; then
    FINAL_STATUS="partial failure (v80=$V80_EXIT, v25=$V25_EXIT)"
fi
log "=== AIST++ medium queue complete: $FINAL_STATUS ==="
log "Per-run logs: $LOG_DIR/v80.log, $LOG_DIR/v25.log"

if [[ "$V80_EXIT" -ne 0 || "$V25_EXIT" -ne 0 ]]; then
    exit 1
fi
