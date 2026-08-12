#!/usr/bin/env bash
# Queue helper for v81 medium run on A800 GPU 4.
# Polls until GPU 4 is free, then launches the v81 medium run.
#
# Usage
# -----
#   nohup bash scripts/queue_v81_true_gt_h36m_medium_a800.sh > outputs/ablations/v81_true_gt_h36m_medium_a800.queue.log 2>&1 &

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

TARGET_GPU=4
INTERVAL_SEC=${INTERVAL_SEC:-60}
LOG=outputs/ablations/v81_true_gt_h36m_medium_a800.queue.log

mkdir -p outputs/ablations

echo "$(date -Iseconds) Queueing v81 medium run on GPU ${TARGET_GPU}..." >> "$LOG"

while true; do
    # Check if GPU 4 is essentially idle (no process using > 1000 MiB and low utilization).
    # We consider it free when utilization is < 10% and memory used < 1000 MiB.
    read -r utilization memory_used <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "${TARGET_GPU}" | sed 's/ //g' | tr ',' ' ')"

    if [[ "${utilization%%}" -lt 10 ]] && [[ "${memory_used% MiB}" -lt 1000 ]]; then
        echo "$(date -Iseconds) GPU ${TARGET_GPU} is free (util=${utilization}, mem=${memory_used}). Launching v81 medium run." >> "$LOG"
        bash scripts/run_v81_true_gt_h36m_medium_a800.sh >> "$LOG" 2>&1
        echo "$(date -Iseconds) v81 medium run finished." >> "$LOG"
        exit 0
    fi

    echo "$(date -Iseconds) GPU ${TARGET_GPU} still busy (util=${utilization}, mem=${memory_used}). Waiting ${INTERVAL_SEC}s..." >> "$LOG"
    sleep "$INTERVAL_SEC"
done
