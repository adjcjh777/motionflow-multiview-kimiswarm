#!/usr/bin/env bash
# Post-training evaluation suite for the v85 random view dropout medium run.
#
# This script:
#   1. Waits until the v85 training job is no longer running.
#   2. Waits for the first free A800 project GPU (6 or 7).
#   3. Runs the standard H36M true-GT test-set evaluation.
#   4. Waits for another free GPU, then runs the no-fallback variable-view eval.
#   5. Waits for another free GPU, then runs the DLT-fallback variable-view eval.
#
# GPU policy: MotionFlow-MultiView only uses GPUs 6 and 7 on A800.  GPUs 0-5 are
# reserved for other projects and must NOT be touched.
#
# Intended to be launched after v85 training finishes; it does not start any
# eval until the training process exits.
#
# Usage:
#   bash scripts/run_v85_post_training_eval_suite.sh

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p "outputs/sota_baselines"
LOG="outputs/sota_baselines/run_v85_post_training_eval_suite.log"

# Redirect stdout/stderr to both the log and the caller's console.
exec > >(tee -a "${LOG}")
exec 2>&1

CHECKPOINT="outputs/ablations/v85_random_view_dropout_medium_a800.pth"

# ---------------------------------------------------------------------------
# Helper: return 0 if the v85 training process is still running.
# We identify it by matching the v85-specific command line; this is more robust
# than hard-coding a PID.
# ---------------------------------------------------------------------------
is_v85_training_running() {
    local pids
    pids=$(pgrep -f 'train_omniview_fusion_v5_webbridge_multi.py.*v85_dropout_prob' || true)
    [[ -n "${pids}" ]]
}

# ---------------------------------------------------------------------------
# Helper: echo the first free GPU among {6,7} (memory used < 1000 MiB).
# Returns non-zero if neither GPU is free.
# ---------------------------------------------------------------------------
select_free_gpu() {
    local i used
    for i in 6 7; do
        used=$(nvidia-smi --id="${i}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
        if (( used < 1000 )); then
            echo "${i}"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Helper: run a single eval stage, waiting for a free GPU first.
# Arguments:
#   $1 stage name (for logging)
#   $2 path to the eval script to run
#   $3 log file for this stage
# ---------------------------------------------------------------------------
run_eval_stage() {
    local stage_name="$1"
    local script_path="$2"
    local log_path="$3"
    local free_gpu pid

    while true; do
        free_gpu=$(select_free_gpu) && break
        echo "[$(date -Iseconds)] No free GPU for ${stage_name}, waiting..."
        sleep 60
    done

    export CUDA_VISIBLE_DEVICES="${free_gpu}"
    echo "[$(date -Iseconds)] ${stage_name}: using GPU ${free_gpu}"

    nohup bash "${script_path}" > "${log_path}" 2>&1 &
    pid=$!
    echo "[$(date -Iseconds)] ${stage_name}: launched PID ${pid}"

    wait "${pid}"
    echo "[$(date -Iseconds)] ${stage_name}: complete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] v85 post-training eval suite started"

# 1. Wait for v85 training to finish.
echo "[$(date -Iseconds)] Waiting for v85 training to finish..."
while is_v85_training_running; do
    echo "[$(date -Iseconds)] v85 training still running, waiting..."
    sleep 60
done
echo "[$(date -Iseconds)] v85 training is no longer running"

# Sanity check: the final checkpoint should exist.  If not, warn but continue;
# the individual eval scripts will error out explicitly if the checkpoint is missing.
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[$(date -Iseconds)] WARNING: expected checkpoint not found: ${CHECKPOINT}"
fi

# 2. Standard H36M true-GT test-set evaluation.
run_eval_stage \
    "v85 test-set eval" \
    "scripts/run_eval_v85_random_view_dropout_medium_a800.sh" \
    "outputs/eval_v85_random_view_dropout_medium_a800_nohup.log"

# 3. No-fallback variable-view evaluation (k=2/3/4).
run_eval_stage \
    "v85 no-fallback variable-view eval" \
    "scripts/eval_variable_views_v85_random_view_dropout_medium_a800.sh" \
    "outputs/variable_view_v85_random_view_dropout_medium_a800_nohup.log"

# 4. DLT-fallback variable-view evaluation (k=2/3/4).
run_eval_stage \
    "v85 DLT-fallback variable-view eval" \
    "scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh" \
    "outputs/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback_nohup.log"

echo "[$(date -Iseconds)] v85 post-training eval suite complete"
