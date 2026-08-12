#!/usr/bin/env bash
# Queue wrapper: wait until A800 GPU 4 is free, then launch
# scripts/run_v25_mixed_relaunch_a800_gpuX.sh with GPU 4.
#
# Usage:
#   nohup bash scripts/queue_v25_mixed_relaunch_a800_gpu4.sh &

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-a800-D}
REMOTE_REPO=${REMOTE_REPO:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20}
GPU=4
POLL_INTERVAL=60

echo "Queueing v25 mixed-dataset relaunch on A800 GPU ${GPU}..."
echo "Polling every ${POLL_INTERVAL}s until GPU ${GPU} has no compute processes."

while true; do
    # Get list of compute processes on the target GPU.
    mapfile -t proc_lines < <(ssh "${REMOTE_HOST}" "nvidia-smi --id=${GPU} --query-compute-apps=pid,process_name --format=csv,noheader,nounits" || true)

    # Remove a possible trailing empty line caused by nvidia-smi output.
    proc_lines=("${proc_lines[@]:-}")

    if [ "${#proc_lines[@]}" -eq 0 ] || [ "${#proc_lines[@]}" -eq 1 ] && [ -z "${proc_lines[0]:-}" ]; then
        echo "[$(date -Iseconds)] GPU ${GPU} is free; launching v25 mixed-dataset relaunch..."
        ssh "${REMOTE_HOST}" "cd ${REMOTE_REPO} && bash scripts/run_v25_mixed_relaunch_a800_gpuX.sh ${GPU}"
        echo "[$(date -Iseconds)] Launch command submitted."
        break
    fi

    echo "[$(date -Iseconds)] GPU ${GPU} still busy: ${#proc_lines[@]} compute process(es). Sleeping ${POLL_INTERVAL}s..."
    sleep "${POLL_INTERVAL}"
done
