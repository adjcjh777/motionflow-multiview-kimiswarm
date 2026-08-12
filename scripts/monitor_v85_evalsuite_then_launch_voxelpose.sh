#!/usr/bin/env bash
# Wait for the v85 post-training evaluation suite to finish, then auto-launch
# VoxelPose training on the first free A800 GPU (GPU 6 or 7 only).
#
# This monitor supersedes scripts/monitor_v85_then_launch_voxelpose.sh, which
# only watched the old no-fallback variable-view eval PID.
#
# Usage:
#   bash scripts/monitor_v85_evalsuite_then_launch_voxelpose.sh [EVAL_SUITE_PID]
#
# Example:
#   nohup bash scripts/monitor_v85_evalsuite_then_launch_voxelpose.sh 2072251 \
#       > outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose_nohup.log 2>&1 &
#
# Important:
#   - Only GPUs 6 and 7 may be used (project policy).
#   - This script intentionally does NOT launch anything until the v85 training
#     and its full post-training eval suite have terminated and a GPU is free.
#   - It is safe to start this while v85/eval suite is still running.

set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines

LOG="outputs/sota_baselines/monitor_v85_evalsuite_then_launch_voxelpose.log"
exec > >(tee -a "${LOG}")
exec 2>&1

# PID of the eval-suite monitor (default to the currently known PID).
: "${1:-2072251}"
EVAL_SUITE_PID="${1}"

echo "[$(date -Iseconds)] VoxelPose auto-launch monitor starting"
echo "[$(date -Iseconds)] Waiting for v85 post-training eval suite monitor PID ${EVAL_SUITE_PID}"

# ---------------------------------------------------------------------------
# 1. Wait for the v85 post-training eval-suite monitor to finish.
# ---------------------------------------------------------------------------
while true; do
    if ! ps -p "${EVAL_SUITE_PID}" >/dev/null 2>&1; then
        echo "[$(date -Iseconds)] v85 post-training eval suite monitor (PID ${EVAL_SUITE_PID}) no longer running"
        break
    fi
    echo "[$(date -Iseconds)] v85 eval suite monitor still running, waiting..."
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

echo "[$(date -Iseconds)] GPU ${FREE_GPU} is free; launching VoxelPose"

# ---------------------------------------------------------------------------
# 3. Launch VoxelPose training on the free GPU.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
nohup bash scripts/run_voxelpose_h36m_true_gt_a800.sh \
    > outputs/sota_baselines/voxelpose_h36m_true_gt_a800_nohup_auto.log 2>&1 &

echo "[$(date -Iseconds)] VoxelPose launched on GPU ${FREE_GPU}; PID $!"
