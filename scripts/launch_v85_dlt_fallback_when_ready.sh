#!/usr/bin/env bash
# Wait for the v85 no-fallback variable-view eval to finish, then launch the
# DLT-fallback variable-view eval on the first available A800 GPU (6 or 7).
# This script does not block the no-fallback eval; it simply queues the fallback.
set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

LOG="outputs/variable_view_fix/launch_v85_dlt_fallback_when_ready.log"
mkdir -p "outputs/variable_view_fix"

# Try to find the running no-fallback eval process.
NOFB_PID=$(pgrep -f 'experiments/eval_variable_views.py.*v85_random_view_dropout_medium_a800\.pth' || true)

if [[ -n "${NOFB_PID}" ]]; then
    echo "[$(date -Iseconds)] Found v85 no-fallback eval PID(s): ${NOFB_PID}" | tee -a "${LOG}"
    echo "[$(date -Iseconds)] Waiting for it to finish before launching DLT-fallback..." | tee -a "${LOG}"
    while ps -p ${NOFB_PID} >/dev/null 2>&1; do
        sleep 60
    done
    echo "[$(date -Iseconds)] v85 no-fallback eval finished." | tee -a "${LOG}"
else
    echo "[$(date -Iseconds)] No running v85 no-fallback eval found." | tee -a "${LOG}"
fi

# Wait for either GPU 6 or 7 to be free (< 1000 MiB used).
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
    echo "[$(date -Iseconds)] No free A800 GPU (allowed: 6 or 7), waiting..." | tee -a "${LOG}"
    sleep 60
done

echo "[$(date -Iseconds)] GPU ${FREE_GPU} is free, launching v85 DLT-fallback eval" | tee -a "${LOG}"

export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
bash scripts/eval_variable_views_v85_random_view_dropout_medium_a800_dlt_fallback.sh

echo "[$(date -Iseconds)] v85 DLT-fallback eval launched." | tee -a "${LOG}"
