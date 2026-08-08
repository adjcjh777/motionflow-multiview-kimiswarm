#!/usr/bin/env bash
# Launch scripts/monitor_v25_local_4090.sh via nohup so it survives shell disconnects.
set -euo pipefail

LOG=${LOG:-outputs/v25_local_4090_nohup.log}
mkdir -p outputs
rm -f "$LOG"

echo "[$(date)] Launching v25 local 4090 monitor via nohup..."
nohup bash scripts/monitor_v25_local_4090.sh > "$LOG" 2>&1 &
PID=$!
echo "$PID" > outputs/v25_local_4090_nohup.pid
echo "Started background monitor PID=$PID"
echo "View log with: tail -f $LOG"
