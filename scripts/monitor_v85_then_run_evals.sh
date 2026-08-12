#!/usr/bin/env bash
# Monitor the v85 random view dropout training and launch the full
# evaluation suite once it finishes.
set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines

LOG="outputs/sota_baselines/monitor_v85_then_run_evals.log"
exec > >(tee -a "${LOG}")
exec 2>&1

echo "[$(date -Iseconds)] Monitoring v85 training (PID 2058225)"

# Wait for the v85 training process to finish.
while true; do
    if ! ps -p 2058225 >/dev/null 2>&1; then
        echo "[$(date -Iseconds)] v85 training (PID 2058225) no longer running"
        break
    fi
    echo "[$(date -Iseconds)] v85 training still running, waiting..."
    sleep 60
done

# Wait until either GPU 6 or 7 is free (memory used < 1000 MiB).
select_free_gpu() {
    local i
    for i in 6 7; do
        local used
        used=$(nvidia-smi --id="${i}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
        if (( used < 1000 )); then
            echo "${i}"
            return 0
        fi
    done
    return 1
}

while true; do
    FREE_GPU=$(select_free_gpu) && break
    echo "[$(date -Iseconds)] No free GPU on A800 (allowed: 6 or 7), waiting..."
    sleep 60
done

echo "[$(date -Iseconds)] GPU ${FREE_GPU} is free, launching v85 evaluation suite"

# 1. Standard test-set evaluation.
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
nohup bash scripts/run_eval_v85_random_view_dropout_medium_a800.sh > outputs/eval_v85_random_view_dropout_medium_a800_nohup.log 2>&1 &
EVAL_PID=$!
echo "[$(date -Iseconds)] Launched v85 test eval PID ${EVAL_PID}"

# Wait for the test eval to finish before occupying both GPUs.
while ps -p ${EVAL_PID} >/dev/null 2>&1; do
    echo "[$(date -Iseconds)] v85 test eval still running, waiting..."
    sleep 60
done

# 2. Variable-view no-fallback eval.
FREE_GPU=$(select_free_gpu) || {
    echo "[$(date -Iseconds)] No free GPU for v85 no-fallback eval, aborting remaining evals"
    exit 1
}
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
nohup bash scripts/eval_variable_views_v85_random_view_dropout_medium_a800.sh > outputs/variable_view_v85_random_view_dropout_medium_a800_nohup.log 2>&1 &
NOFB_PID=$!
echo "[$(date -Iseconds)] Launched v85 no-fallback variable-view eval PID ${NOFB_PID}"

while ps -p ${NOFB_PID} >/dev/null 2>&1; do
    echo "[$(date -Iseconds)] v85 no-fallback eval still running, waiting..."
    sleep 60
done

# 3. Variable-view DLT-fallback eval.
FREE_GPU=$(select_free_gpu) || {
    echo "[$(date -Iseconds)] No free GPU for v85 DLT-fallback eval, aborting"
    exit 1
}
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
nohup bash scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh > outputs/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback_nohup.log 2>&1 &
DLT_PID=$!
echo "[$(date -Iseconds)] Launched v85 DLT-fallback variable-view eval PID ${DLT_PID}"

while ps -p ${DLT_PID} >/dev/null 2>&1; do
    echo "[$(date -Iseconds)] v85 DLT-fallback eval still running, waiting..."
    sleep 60
done

echo "[$(date -Iseconds)] v85 evaluation suite complete"
