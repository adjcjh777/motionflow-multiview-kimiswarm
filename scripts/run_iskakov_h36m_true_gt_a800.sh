#!/usr/bin/env bash
# Queue an Iskakov ICCV 2019 learnable-triangulation H36M true-GT baseline
# run on A800 after the v85 random-view-dropout training finishes.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# This monitor waits for v85 (PID 2058225) to terminate, then picks the first
# free GPU among {6,7} and launches the Iskakov baseline there.
#
# Usage:
#   nohup bash scripts/run_iskakov_h36m_true_gt_a800.sh \
#       > outputs/sota_baselines/iskakov_h36m_true_gt_a800_queued_nohup.log 2>&1 &
#
# The script can be started while v85 is still running.

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines outputs/baselines

LOG="outputs/sota_baselines/iskakov_h36m_true_gt_a800_queued.log"
exec > >(tee -a "${LOG}")
exec 2>&1

V85_PID="${V85_PID:-2058225}"

echo "[$(date -Iseconds)] Iskakov H36M true-GT A800 queue monitor starting"
echo "[$(date -Iseconds)] Will wait for v85 training (PID ${V85_PID}) to finish"

# ---------------------------------------------------------------------------
# 1. Wait for the v85 training process to finish.
# ---------------------------------------------------------------------------
while true; do
    if ! ps -p "${V85_PID}" >/dev/null 2>&1; then
        echo "[$(date -Iseconds)] v85 training (PID ${V85_PID}) no longer running"
        break
    fi
    echo "[$(date -Iseconds)] v85 training still running, waiting..."
    sleep 60
done

# Give any child processes a moment to release GPU memory.
sleep 10

# ---------------------------------------------------------------------------
# 2. Wait until either GPU 6 or 7 is free (memory used < 1000 MiB).
# ---------------------------------------------------------------------------
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

echo "[$(date -Iseconds)] Waiting for a free GPU on A800 (allowed: 6 or 7)"

while true; do
    FREE_GPU=$(select_free_gpu) && break
    echo "[$(date -Iseconds)] No free GPU on A800 (allowed: 6 or 7), waiting..."
    sleep 60
done

echo "[$(date -Iseconds)] GPU ${FREE_GPU} is free; launching Iskakov H36M true-GT baseline"

# ---------------------------------------------------------------------------
# 3. Launch Iskakov ICCV 2019 H36M true-GT baseline on the free GPU.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

nohup "${PYTHON}" -u experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m \
    --epochs 10 \
    --train_samples_per_epoch 4096 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 8 \
    --ref_max_frames 2000 \
    --log_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800.log \
    --ckpt_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800.pth \
    > outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_nohup.log 2>&1 &

PID=$!
echo "[$(date -Iseconds)] Launched Iskakov H36M true-GT baseline on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "[$(date -Iseconds)] Log: outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800.log"
