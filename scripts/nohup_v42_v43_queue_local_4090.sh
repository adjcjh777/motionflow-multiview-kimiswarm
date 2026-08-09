#!/usr/bin/env bash
# nohup queue: run v42 full, then v43 full, then v43 quick ablation.
# Keeps the local RTX 4090 busy even if the shell session disconnects.
set -euo pipefail

LOG="outputs/nohup_v42_v43_queue_local_4090.log"
nohup bash -c '
  echo "[$(date)] Starting v42 full..." >> '"$LOG"'
  bash scripts/run_v42_full_local_4090.sh
  echo "[$(date)] v42 full done." >> '"$LOG"'
  echo "[$(date)] Starting v43 full..." >> '"$LOG"'
  bash scripts/run_v43_full_local_4090.sh
  echo "[$(date)] v43 full done." >> '"$LOG"'
  echo "[$(date)] Starting v43 quick ablation..." >> '"$LOG"'
  bash scripts/run_v43_quick_local_4090.sh
  echo "[$(date)] v43 quick ablation done." >> '"$LOG"'
' >> "$LOG" 2>&1 &
disown
echo "[$(date)] Local v42 -> v43 nohup queue launched. Log: $LOG"
