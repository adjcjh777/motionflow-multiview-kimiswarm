#!/usr/bin/env bash
# Start a nohup-based monitor for a local training log.
# Usage: bash scripts/nohup_monitor_local_val_ready.sh outputs/omniview_fusion_v26_udp_full_local_4090.log
set -euo pipefail

LOG_FILE="${1:-outputs/omniview_fusion_v26_udp_full_local_4090.log}"
BASENAME=$(basename "$LOG_FILE" .log)
MONITOR_LOG="outputs/monitor_${BASENAME}.log"
PID_FILE="outputs/monitor_${BASENAME}.pid"

mkdir -p outputs
rm -f "$MONITOR_LOG" "$PID_FILE"

echo "Starting nohup monitor for $LOG_FILE"
nohup python -u scripts/monitor_local_val_ready.py --log "$LOG_FILE" --poll-sec 60 > "$MONITOR_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Monitor PID=$(cat "$PID_FILE"), log=$MONITOR_LOG"
