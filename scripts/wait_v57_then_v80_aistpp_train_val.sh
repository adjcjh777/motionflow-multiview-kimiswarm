#!/usr/bin/env bash
# Wait until the local RTX 4090 is idle and no v57 H36M true-GT medium process
# is running, then start v80 AIST++ train/val training on the local RTX 4090.
#
# This wrapper enforces the project rule of at most ONE training task on the
# local GPU at a time. It is prepared but NOT executed.
#
# Usage (manual, once v57 has finished):
#   bash scripts/wait_v57_then_v80_aistpp_train_val.sh
#
# To run detached:
#   nohup bash scripts/wait_v57_then_v80_aistpp_train_val.sh \
#       > outputs/wait_v57_then_v80_aistpp_train_val_nohup.log 2>&1 &
set -euo pipefail

POLL_SEC=${POLL_SEC:-60}
LOG_FILE="outputs/wait_v57_then_v80_aistpp_train_val_$(date +%Y%m%d_%H%M%S).log"

V57_SCRIPT="scripts/run_v57_h36m_true_gt_medium.sh"
V57_MARKER="omniview_fusion_v57_h36m_true_gt_medium"
V80_SCRIPT="scripts/run_v80_aistpp_train_val_local_4090.sh"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Return 0 if a v57 H36M true-GT medium process is still running.
is_v57_running() {
    ps -ef 2>/dev/null \
        | grep -v grep \
        | grep -E "${V57_SCRIPT}|${V57_MARKER}" \
        >/dev/null 2>&1
}

# Return 0 if there are no python compute processes on the GPU.
is_gpu_idle() {
    local count
    count=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -i "python" \
        | wc -l)
    [[ "$count" -eq 0 ]]
}

log "Waiting for v57 H36M true-GT medium training to finish and GPU to become idle..."
while true; do
    v57_busy=0
    gpu_busy=0

    if is_v57_running; then
        v57_busy=1
    fi

    if ! is_gpu_idle; then
        gpu_busy=1
    fi

    if [[ "$v57_busy" -eq 0 && "$gpu_busy" -eq 0 ]]; then
        log "v57 finished and GPU is idle."
        break
    fi

    if [[ "$v57_busy" -ne 0 ]]; then
        log "v57 still running."
    fi

    if [[ "$gpu_busy" -ne 0 ]]; then
        log "GPU not idle (python compute process still present)."
    fi

    log "Polling again in ${POLL_SEC}s..."
    sleep "$POLL_SEC"
done

log "Starting v80 AIST++ train/val: $V80_SCRIPT"
bash "$V80_SCRIPT"
log "v80 AIST++ train/val completed."
