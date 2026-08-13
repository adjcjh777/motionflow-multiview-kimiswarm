#!/usr/bin/env bash
# Chain watcher: after the v81 true-GT v2 medium training finishes,
# sequentially launch v82, v46, v52, and v57 true-GT v2 medium trainings
# on the first free project GPU (6 or 7).
#
# Usage
# -----
#   # Run on A800 (recommended: detached with nohup)
#   nohup bash scripts/launch_v82_v46_v52_v57_after_v81.sh \
#       > outputs/launch_v82_v46_v52_v57_after_v81.log 2>&1 &
#
#   # Attach to the current model's tmux session at any time
#   ssh a800-D "tmux attach -t <MODEL>_true_gt_v2_medium_a800"

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
POLL_SEC=60
LOG_FILE="${REPO}/outputs/launch_v82_v46_v52_v57_after_v81.log"

# GPU policy: MotionFlow-MultiView only uses GPUs 6 and 7 on A800.
ALLOWED_GPUS=(6 7)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Wait for the tmux session to appear (v81 hasn't started yet).
wait_for_session_start() {
    local session="$1"
    while ! tmux has-session -t "${session}" 2>/dev/null; do
        log "Session '${session}' not yet started; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done
}

# Wait until the named tmux session no longer exists.
wait_for_session_end() {
    local session="$1"
    while tmux has-session -t "${session}" 2>/dev/null; do
        log "Session '${session}' still active; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done
}

# Return the first GPU index (6 or 7) whose utilization and memory are below
# threshold and which has no compute processes. Empty string if none free.
find_free_gpu() {
    local gpu util mem procs
    for gpu in "${ALLOWED_GPUS[@]}"; do
        read -r util mem <<< "$(nvidia-smi --id=${gpu} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')"
        procs="$(nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"

        # Strip units and default to "busy" if values are empty.
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
    echo ""
}

# Wait for a free project GPU, then launch a training run in its own tmux session.
launch_when_free() {
    local target_script="$1"
    local tmux_name="$2"
    local free_gpu

    log "Waiting for a free project GPU to launch ${target_script}..."
    while true; do
        free_gpu="$(find_free_gpu)"
        if [[ -n "${free_gpu}" ]]; then
            log "GPU ${free_gpu} is free. Launching ${target_script} in tmux '${tmux_name}'."
            tmux kill-session -t "${tmux_name}" 2>/dev/null || true
            tmux new-session -d -s "${tmux_name}" \
                "cd ${REPO} && CUDA_VISIBLE_DEVICES=${free_gpu} bash ${target_script}"
            log "Launched ${target_script} on GPU ${free_gpu}."
            return 0
        fi
        log "No free project GPU; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done
}

cd "${REPO}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

log "Chain watcher started. Will run v82 -> v46 -> v52 -> v57 after v81."

# 1) Wait for v81 to start, then finish.
wait_for_session_start "v81_true_gt_v2_medium_a800"
wait_for_session_end "v81_true_gt_v2_medium_a800"
log "v81 session ended. Starting v82."

# 2) v82 multi-scale temporal-pose-attention.
launch_when_free "scripts/run_v82_true_gt_v2_medium_a800.sh" "v82_true_gt_v2_medium_a800"
wait_for_session_end "v82_true_gt_v2_medium_a800"
log "v82 session ended. Starting v46."

# 3) v46 Sparse-View Geometry (SVG).
launch_when_free "scripts/run_v46_true_gt_v2_medium_a800.sh" "v46_true_gt_v2_medium_a800"
wait_for_session_end "v46_true_gt_v2_medium_a800"
log "v46 session ended. Starting v52."

# 4) v52 Uncertainty-Weighted Triangulation (UWT).
launch_when_free "scripts/run_v52_true_gt_v2_medium_a800.sh" "v52_true_gt_v2_medium_a800"
wait_for_session_end "v52_true_gt_v2_medium_a800"
log "v52 session ended. Starting v57."

# 5) v57 Domain-Conditional Physical-Space Calibration (DC-PSC).
launch_when_free "scripts/run_v57_true_gt_v2_medium_a800.sh" "v57_true_gt_v2_medium_a800"
wait_for_session_end "v57_true_gt_v2_medium_a800"

log "Chain complete: v81 -> v82 -> v46 -> v52 -> v57 all finished."
