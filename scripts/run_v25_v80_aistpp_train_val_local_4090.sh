#!/usr/bin/env bash
# Sequential wrapper for AIST++ train/val on the local RTX 4090.
#
# Usage:
#   bash scripts/run_v25_v80_aistpp_train_val_local_4090.sh
#
# Optional: run v80 first, then v25:
#   RUN_V80_FIRST=1 bash scripts/run_v25_v80_aistpp_train_val_local_4090.sh
#
# Rules enforced by this wrapper:
#   * Polls nvidia-smi/processes and waits until the GPU is idle before
#     launching any training script.
#   * If RUN_V80_FIRST=1, v80 AIST++ train/val is launched first and v25 is
#     queued to run only after v80 finishes (and the GPU is free again).
#   * Otherwise only v25 AIST++ train/val is launched once the GPU is free.
#
# This wrapper never runs anything on A800-D and will refuse to start there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_GPU="${SCRIPT_DIR}/sota_baselines/check_gpu_free.sh"
RUN_V80_FIRST="${RUN_V80_FIRST:-0}"

LOG_DIR="outputs/aistpp_train_val_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue.log"

wait_for_gpu() {
  # Guard against the helper being missing.
  if [[ ! -f "$CHECK_GPU" ]]; then
    echo "[$(date)] ERROR: GPU check helper missing: $CHECK_GPU" | tee -a "$QUEUE_LOG" >&2
    exit 1
  fi

  while true; do
    if bash "$CHECK_GPU" >"${LOG_DIR}/gpu_check.log" 2>&1; then
      echo "[$(date)] GPU is idle." | tee -a "$QUEUE_LOG"
      return 0
    fi
    echo "[$(date)] GPU is busy, waiting... (see ${LOG_DIR}/gpu_check.log)" | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

run_v80() {
  echo "[$(date)] === Queueing v80 AIST++ train/val ===" | tee -a "$QUEUE_LOG"
  wait_for_gpu
  echo "[$(date)] Starting v80 AIST++ train/val" | tee -a "$QUEUE_LOG"
  bash "${SCRIPT_DIR}/run_v80_aistpp_train_val_local_4090.sh" >"$LOG_DIR/v80.log" 2>&1
  echo "[$(date)] v80 AIST++ train/val finished (exit $?)" | tee -a "$QUEUE_LOG"
}

run_v25() {
  echo "[$(date)] === Queueing v25 AIST++ train/val ===" | tee -a "$QUEUE_LOG"
  wait_for_gpu
  echo "[$(date)] Starting v25 AIST++ train/val" | tee -a "$QUEUE_LOG"
  bash "${SCRIPT_DIR}/run_v25_aistpp_train_val_local_4090.sh" >"$LOG_DIR/v25.log" 2>&1
  echo "[$(date)] v25 AIST++ train/val finished (exit $?)" | tee -a "$QUEUE_LOG"
}

if [[ "$RUN_V80_FIRST" == "1" ]]; then
  run_v80
  run_v25
else
  run_v25
fi

echo "[$(date)] AIST++ train/val queue complete. Logs: $LOG_DIR" | tee -a "$QUEUE_LOG"
