#!/usr/bin/env bash
# GPU training queue for the local RTX 4090.
#
# Monitors the local GPU and waits for the running v25 medium baseline to
# finish and for the GPU to become idle. Then it runs the v80 and v57 medium
# baselines back-to-back.
#
# Usage:
#   bash scripts/gpu_queue.sh
#
# To run detached:
#   nohup bash scripts/gpu_queue.sh > outputs/gpu_queue_nohup.log 2>&1 &
#
# The script obeys the project rule of at most ONE training task on the local
# RTX 4090 at a time.
set -euo pipefail

# Configuration
V25_MARKER="omniview_fusion_v25_h36m_true_gt_medium"
V25_SCRIPT="run_v25_h36m_true_gt_medium_local_4090.sh"
V80_SCRIPT="scripts/run_v80_h36m_true_gt_medium.sh"
V57_SCRIPT="scripts/run_v57_h36m_true_gt_medium.sh"
POLL_SEC=${GPU_QUEUE_POLL_SEC:-60}

# Logging
LOG_DIR="outputs/gpu_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/gpu_queue.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

# Return 0 if the v25 medium training process is still running.
is_v25_running() {
    # Match either the explicit v25 medium script name or the output path marker
    # in the python training command line.
    ps -ef 2>/dev/null \
        | grep -v grep \
        | grep -E "${V25_SCRIPT}|${V25_MARKER}" \
        >/dev/null 2>&1
}

# Return 0 if there are no python training processes on the GPU.
# On WSL/Windows nvidia-smi lists every process that has a context on the
# display driver (desktop, browsers, etc.), so we only count processes whose
# image name contains "python".
is_gpu_idle() {
    local count
    count=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -i "python" \
        | wc -l)
    [[ "$count" -eq 0 ]]
}

# Wait until v25 is gone and nvidia-smi reports no compute processes.
wait_for_idle() {
    log "Waiting for v25 medium to finish and GPU to become idle..."
    while true; do
        local v25_busy=0
        local gpu_busy=0

        if is_v25_running; then
            v25_busy=1
        fi

        if ! is_gpu_idle; then
            gpu_busy=1
        fi

        if [[ "$v25_busy" -eq 0 && "$gpu_busy" -eq 0 ]]; then
            log "v25 medium finished and GPU is idle."
            break
        fi

        if [[ "$v25_busy" -ne 0 ]]; then
            log "v25 medium still running."
        fi

        if [[ "$gpu_busy" -ne 0 ]]; then
            log "GPU not idle (compute process still present)."
        fi

        log "Polling again in ${POLL_SEC}s..."
        sleep "$POLL_SEC"
    done
}

log "GPU queue started."
log "Planned runs: $V80_SCRIPT -> $V57_SCRIPT"
log "Poll interval: ${POLL_SEC}s"
log "Queue log: $QUEUE_LOG"

# Safety wait before starting anything.
wait_for_idle

log "Starting v80 medium: $V80_SCRIPT"
bash "$V80_SCRIPT"
log "v80 medium completed."

# v57 should only start after v80 has fully released the GPU. Re-check idle.
wait_for_idle

log "Starting v57 medium: $V57_SCRIPT"
bash "$V57_SCRIPT"
log "v57 medium completed."

log "GPU queue complete."
