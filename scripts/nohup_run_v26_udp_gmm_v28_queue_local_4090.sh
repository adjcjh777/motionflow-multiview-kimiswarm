#!/usr/bin/env bash
# Launch the v26+UDP-GMM/v28 local 4090 queue via nohup.
set -euo pipefail

LOG="outputs/v26_udp_gmm_v28_queue_local_4090_nohup.log"
mkdir -p outputs
rm -f "$LOG"

echo "[$(date)] Launching v26+UDP-GMM/GMM+v28/v28 queue in the background via nohup..."
nohup bash scripts/run_v26_udp_gmm_v28_queue_local_4090.sh > "$LOG" 2>&1 &

PID=$!
echo "Started background process PID=$PID"
echo "$PID" > outputs/v26_udp_gmm_v28_queue_local_4090.pid
echo "View log with: tail -f $LOG"
