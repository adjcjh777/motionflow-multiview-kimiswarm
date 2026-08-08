#!/usr/bin/env bash
# Wait for the currently running v25 local 4090 baseline to finish, then start
# the v25 variant queue. This is useful when the baseline was launched manually
# and you want the variants to run back-to-back without manual intervention.
set -euo pipefail

PID_FILE=${1:-outputs/v25_local_4090_train.pid}
LOG_DIR="outputs/v25_variant_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ ! -f "$PID_FILE" ]]; then
    echo "[$(date)] PID file $PID_FILE not found; starting variant queue immediately." | tee -a "$LOG_DIR/queue.log"
    bash scripts/run_v25_variant_queue_local_4090.sh > "$LOG_DIR/queue.log" 2>&1
    exit 0
fi

PID=$(cat "$PID_FILE")
echo "[$(date)] Waiting for v25 baseline PID=$PID to finish..." | tee -a "$LOG_DIR/queue.log"

# Wait until the process is gone (Windows PID, so use wmic).
while wmic process where "ProcessId=$PID" get ProcessId /format:csv 2>/dev/null | grep -q "$PID"; do
    sleep 60
done

echo "[$(date)] v25 baseline finished. Starting variant queue." | tee -a "$LOG_DIR/queue.log"
bash scripts/run_v25_variant_queue_local_4090.sh > "$LOG_DIR/queue.log" 2>&1
