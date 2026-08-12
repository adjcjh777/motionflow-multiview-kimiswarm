#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
DETECTED_DIR="$REMOTE_ROOT/data/webbridge/mpi_inf_3dhp_detected_2d"
DLT_JSON="$REMOTE_ROOT/outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

LOG_FILE="$LOCAL_REPO/outputs/monitor_mpi_completion.log"
mkdir -p "$LOCAL_REPO/outputs"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Starting MPI-INF-3DHP completion monitor"

last_count=0
while true; do
    count=$(ssh a800-D "ls $DETECTED_DIR/*.npz 2>/dev/null | wc -l" || echo 0)
    if [ "$count" -ne "$last_count" ]; then
        echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] $count/16 .npz files ready"
        last_count=$count
    fi
    if [ "$count" -ge 16 ]; then
        echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] All 16 .npz files ready; waiting for DLT baseline JSON..."
        break
    fi
    sleep 300
done

# Wait for the DLT JSON to contain 16 files
while true; do
    ready=$(ssh a800-D "python3 -c \"import json; d=json.load(open('$DLT_JSON')); print(len(d.get('per_file', [])))\"" || echo 0)
    if [ "$ready" -ge 16 ]; then
        echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] DLT baseline JSON has $ready entries"
        break
    fi
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] DLT JSON has $ready entries, waiting..."
    sleep 300
done

# Fetch numbers
ssh a800-D "cat $DLT_JSON" > "$LOCAL_REPO/outputs/mpi_dlt_baseline_detected_2d.json"
python "$LOCAL_REPO/scripts/record_mpi_dlt_result.py"

echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Done"
