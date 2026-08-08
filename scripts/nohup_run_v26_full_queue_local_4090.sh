#!/usr/bin/env bash
# Launch the full-scale v26+UDP/GMM/v28 local 4090 queue via nohup so it survives
# the current shell session (fallback because tmux is not available in this WSL env).
set -euo pipefail

LOG="outputs/v26_full_queue_local_4090_nohup.log"
mkdir -p outputs

# Remove any previous nohup log to avoid confusion.
rm -f outputs/v26_full_queue_local_4090_nohup.log

echo "[$(date)] Launching full v26+UDP/GMM/v28 queue in the background via nohup..."
nohup bash scripts/run_v26_full_queue_local_4090.sh > "$LOG" 2>&1 &

PID=$!
echo "Started background process PID=$PID"
echo "View log with: tail -f $LOG"

# Write a small status file with the PID for later checks.
echo "$PID" > outputs/v26_full_queue_local_4090.pid
