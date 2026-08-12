#!/usr/bin/env bash
# Run the three v25 H36M true-GT divergence ablations one after another.
#
# The ablations are:
#   1. v25_true_gt_baseline_fix          (hyperparameter fix)
#   2. v25_true_gt_geometry_regularization (bone / joint-limit / temporal-bone constraints)
#   3. v25_true_gt_mixed_dataset         (H36M + MPI-INF-3DHP mixed loader)
#
# Usage
# -----
#   # Dry-run: print commands without executing (default)
#   bash scripts/run_v25_ablations_sequential.sh
#
#   # Poll until the local GPU is free, then run all three ablations sequentially
#   bash scripts/run_v25_ablations_sequential.sh --run
#
#   # Run with a custom polling interval (seconds)
#   POLL_SEC=30 bash scripts/run_v25_ablations_sequential.sh --run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

POLL_SEC=${POLL_SEC:-60}
GRACE_SEC=${GRACE_SEC:-90}
LOG_FILE="outputs/run_v25_ablations_sequential_$(date +%Y%m%d_%H%M%S).log"
RUN=0

# Parse arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--run]" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${REPO_ROOT}/outputs/ablations"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Return 0 if there are no python compute processes on the GPU.
is_gpu_idle() {
    local count
    count=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | grep -i "python" \
        | wc -l)
    [[ "$count" -eq 0 ]]
}

# Wait until the local RTX 4090 is completely idle, with a grace period to
# avoid racing a process that has just exited but has not released memory.
wait_for_idle_gpu() {
    log "Waiting for all python GPU processes to finish..."
    while true; do
        if is_gpu_idle; then
            log "GPU reports no python processes. Grace period ${GRACE_SEC}s..."
            sleep "${GRACE_SEC}"
            if is_gpu_idle; then
                log "GPU still idle after grace period."
                return 0
            else
                log "A new python GPU process appeared during grace period. Continuing to wait."
            fi
        fi
        log "GPU not idle yet. Polling again in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done
}

# Sequential ablation list.
ABLAS=(
    "scripts/run_v25_ablation_true_gt_baseline.sh"
    "scripts/run_v25_ablation_geometry_regularization.sh"
    "scripts/run_v25_ablation_mixed_dataset.sh"
)

if [[ ${RUN} -eq 0 ]]; then
    log "=== v25 true-GT ablations (dry-run) ==="
    for abla in "${ABLAS[@]}"; do
        log "Would run: bash ${abla}"
    done
    log "Dry-run complete. Re-run with --run to execute."
    exit 0
fi

log "=== v25 true-GT ablations (sequential run) ==="
log "Log file: ${LOG_FILE}"

wait_for_idle_gpu

for abla in "${ABLAS[@]}"; do
    log "Starting ablation: ${abla}"
    if bash "${abla}" >> "${LOG_FILE}" 2>&1; then
        log "Ablation finished successfully: ${abla}"
    else
        log "Ablation failed with exit code $?: ${abla}"
        log "Stopping sequential launcher. Fix the issue and re-run."
        exit 1
    fi
    # No need to wait between jobs; the next iteration will find the GPU idle
    # because each ablation script is foreground and blocks until completion.
done

log "All three v25 ablations completed."
