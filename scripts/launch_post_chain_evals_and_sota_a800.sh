#!/usr/bin/env bash
# Post-chain watcher for A800 true-GT v2 leaderboard.
#
# Waits until the v82/v46/v52/v57 chain training finishes, then sequentially
# launches (on the first free project GPU 6/7):
#   1) v81 test-set eval
#   2) v82 test-set eval
#   3) v46 test-set eval
#   4) v52 test-set eval
#   5) v57 test-set eval
#   6) v86 no-fallback variable-view eval
#   7) v86 DLT-fallback variable-view eval
#   8) v85 DLT-fallback variable-view eval
#
# Usage (on A800)
#   nohup bash scripts/launch_post_chain_evals_and_sota_a800.sh \
#       > outputs/post_chain/launch_post_chain_evals_and_sota_a800.log 2>&1 &
#
#   # or inside a tmux session
#   tmux new-session -d -s post_chain_evals \
#       "cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20 && bash scripts/launch_post_chain_evals_and_sota_a800.sh"

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
LOG_FILE="${REPO}/outputs/post_chain/launch_post_chain_evals_and_sota_a800.log"
POLL_SEC=60
ALLOWED_GPUS=(6 7)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Wait until the named tmux session no longer exists.
wait_session_end() {
    local session="$1"
    while tmux has-session -t "${session}" 2>/dev/null; do
        log "Session '${session}' still active; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done
}

# Wait until a training session has finished and its checkpoint exists.
# If the session is not present, fall back to waiting for the checkpoint file.
wait_training_done() {
    local session="$1"
    local ckpt="$2"

    log "Waiting for ${session} (checkpoint: ${ckpt})"

    # Session may already be gone (fast path) or still running.
    while tmux has-session -t "${session}" 2>/dev/null; do
        log "Training session '${session}' still active; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done

    # Wait for the checkpoint to be written (chain watcher saves it after training).
    while [[ ! -f "${ckpt}" ]]; do
        log "Checkpoint ${ckpt} not yet present; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done

    log "Training '${session}' complete and checkpoint present."
}

# Return the first project GPU (6 or 7) that appears free:
# low utilization, low memory, and no compute processes.
find_free_gpu() {
    local gpu util mem procs
    for gpu in "${ALLOWED_GPUS[@]}"; do
        read -r util mem <<< "$(nvidia-smi --id=${gpu} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ')"
        procs="$(nvidia-smi --id=${gpu} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"

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

# Launch a script in a dedicated tmux session on a free GPU, then wait for it to finish.
run_task_on_free_gpu() {
    local script="$1"
    local tmux_name="$2"
    local free_gpu

    log "Waiting for a free project GPU to run ${script}..."
    while true; do
        free_gpu="$(find_free_gpu)"
        if [[ -n "${free_gpu}" ]]; then
            log "GPU ${free_gpu} is free. Launching ${script} in tmux '${tmux_name}'."
            break
        fi
        log "No free project GPU; polling in ${POLL_SEC}s..."
        sleep "${POLL_SEC}"
    done

    # Remove any stale session with the same name, then start fresh.
    tmux kill-session -t "${tmux_name}" 2>/dev/null || true
    tmux new-session -d -s "${tmux_name}" \
        "cd ${REPO} && CUDA_VISIBLE_DEVICES=${free_gpu} bash ${script}"

    log "Launched ${script} on GPU ${free_gpu} in tmux '${tmux_name}'."

    # Block until the eval tmux session exits.
    while tmux has-session -t "${tmux_name}" 2>/dev/null; do
        sleep "${POLL_SEC}"
    done

    log "Eval tmux '${tmux_name}' finished."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cd "${REPO}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

log "Post-chain eval watcher started."

# 1) Wait for the v82 -> v46 -> v52 -> v57 chain to finish.
wait_training_done "v82_true_gt_v2_medium_a800" "${REPO}/outputs/ablations/v82_true_gt_v2_medium_a800.pth"
wait_training_done "v46_true_gt_v2_medium_a800" "${REPO}/outputs/ablations/v46_true_gt_v2_medium_a800.pth"
wait_training_done "v52_true_gt_v2_medium_a800" "${REPO}/outputs/ablations/v52_true_gt_v2_medium_a800.pth"
wait_training_done "v57_true_gt_v2_medium_a800" "${REPO}/outputs/ablations/v57_true_gt_v2_medium_a800.pth"

log "All chain trainings finished. Starting post-chain evals."

# 2) Sequentially run the post-chain eval tasks.
TASKS=(
    "scripts/run_v81_true_gt_v2_test_a800.sh"
    "scripts/run_v82_true_gt_v2_test_a800.sh"
    "scripts/run_v46_true_gt_v2_test_a800.sh"
    "scripts/run_v52_true_gt_v2_test_a800.sh"
    "scripts/run_v57_true_gt_v2_test_a800.sh"
    "scripts/eval_variable_views_v86_no_count_embedding_true_gt_v2_medium_a800.sh"
    "scripts/eval_variable_views_v86_no_count_embedding_true_gt_v2_medium_a800_dlt_fallback.sh"
    "scripts/eval_variable_views_v85_random_view_dropout_true_gt_v2_medium_a800_dlt_fallback.sh"
)

TMUX_NAMES=(
    "post_v81_test"
    "post_v82_test"
    "post_v46_test"
    "post_v52_test"
    "post_v57_test"
    "post_v86_varview"
    "post_v86_dlt_fallback"
    "post_v85_dlt_fallback"
)

for i in "${!TASKS[@]}"; do
    run_task_on_free_gpu "${TASKS[$i]}" "${TMUX_NAMES[$i]}"
done

log "All post-chain evals complete."
