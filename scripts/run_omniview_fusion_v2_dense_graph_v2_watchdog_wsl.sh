#!/usr/bin/env bash
# Watchdog wrapper for the dense+graph v2 full run.
# Restarts the training if the Python process exits non-zero, but stops after
# MAX_RESTARTS so we do not spin forever on a reproducible crash.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/outputs/omniview_fusion_v2_d128_dense_graph_v2.log"
MAX_RESTARTS=3
RESTART=0

# Append a marker so we can see each restart in the log.
echo "[watchdog] starting dense+graph v2 training at $(date -Iseconds)" >> "$LOG"

while true; do
  ((RESTART+=1))
  echo "[watchdog] attempt $RESTART / $MAX_RESTARTS" >> "$LOG"
  bash "${ROOT}/scripts/run_omniview_fusion_v2_dense_graph_v2_full_wsl.sh" >> "$LOG" 2>&1
  EXIT=$?
  echo "[watchdog] python exited with $EXIT at $(date -Iseconds)" >> "$LOG"
  if [ $EXIT -eq 0 ]; then
    echo "[watchdog] training completed successfully" >> "$LOG"
    break
  fi
  if [ $RESTART -ge $MAX_RESTARTS ]; then
    echo "[watchdog] max restarts reached; giving up" >> "$LOG"
    break
  fi
  echo "[watchdog] restarting in 10s..." >> "$LOG"
  sleep 10
done
