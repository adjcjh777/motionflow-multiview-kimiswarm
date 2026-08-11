#!/usr/bin/env bash
# Wait until the local RTX 4090 has no python GPU process, then start
# the v57 H36M true-GT medium run.  Includes a short grace period so a
# process that has just exited is fully gone before v57 tries to claim memory.
set -euo pipefail

POLL_SEC=${POLL_SEC:-60}
GRACE_SEC=${GRACE_SEC:-90}
LOG_FILE="outputs/wait_idle_then_v57_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Count python processes that nvidia-smi reports on the GPU.
python_gpu_count() {
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -ic "python" \
        || true
}

log "Waiting for all python GPU processes to finish before starting v57..."
while true; do
    if [[ "$(python_gpu_count)" -eq 0 ]]; then
        log "GPU reports no python processes. Grace period ${GRACE_SEC}s..."
        sleep "$GRACE_SEC"
        if [[ "$(python_gpu_count)" -eq 0 ]]; then
            log "GPU still idle after grace period. Starting v57 H36M true-GT medium training."
            break
        else
            log "A new python GPU process appeared during grace period. Continuing to wait."
        fi
    fi
    log "GPU not idle yet. Polling again in ${POLL_SEC}s..."
    sleep "$POLL_SEC"
done

bash scripts/run_v57_h36m_true_gt_medium.sh
log "v57 H36M true-GT medium training completed."
