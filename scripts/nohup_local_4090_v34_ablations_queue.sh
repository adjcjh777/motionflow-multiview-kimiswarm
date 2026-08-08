#!/usr/bin/env bash
# Launch the local RTX 4090 v34 ablation queue via nohup.
set -euo pipefail

LOG="outputs/v34_local_4090_ablations_queue_nohup.log"
mkdir -p outputs
rm -f "outputs/v34_local_4090_ablations_queue_nohup.log"

echo "[$(date)] Launching local RTX 4090 v34 ablation queue in the background via nohup..."
nohup python -u scripts/poll_local_4090_queue.py > "$LOG" 2>&1 &

PID=$!
echo "Started background process PID=$PID"
echo "View log with: tail -f $LOG"
echo "$PID" > outputs/v34_local_4090_ablations_queue.pid
