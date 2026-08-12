#!/usr/bin/env bash
# Monitor the v85 no-fallback variable-view eval and launch VoxelPose
# training on A800 GPU 6 once GPU 6 becomes free.
set -euo pipefail

REPO_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
cd "${REPO_ROOT}"

mkdir -p outputs/sota_baselines

LOG="outputs/sota_baselines/monitor_v85_then_launch_voxelpose.log"
exec > >(tee -a "${LOG}")
exec 2>&1

echo "[$(date -Iseconds)] Monitoring v85 no-fallback eval and GPU 6 availability"

# Wait for the v85 no-fallback eval process to finish.
while true; do
    if ! ps -p 2062181 >/dev/null 2>&1; then
        echo "[$(date -Iseconds)] v85 no-fallback eval (PID 2062181) no longer running"
        break
    fi
    echo "[$(date -Iseconds)] v85 no-fallback eval still running, waiting..."
    sleep 60
done

# Wait until GPU 6 is free (memory used < 1000 MiB).
while true; do
    USED=$(nvidia-smi --id=6 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
    echo "[$(date -Iseconds)] GPU 6 memory.used=${USED} MiB"
    if (( USED < 1000 )); then
        break
    fi
    sleep 60
done

echo "[$(date -Iseconds)] GPU 6 is free, launching VoxelPose training"

# Launch VoxelPose on GPU 6.
export CUDA_VISIBLE_DEVICES=6
nohup bash scripts/run_voxelpose_h36m_true_gt_a800.sh > outputs/sota_baselines/voxelpose_h36m_true_gt_a800_nohup3.log 2>&1 &

echo "[$(date -Iseconds)] VoxelPose launched PID $!"
