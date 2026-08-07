#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
INTERVAL=60
LOG=$ROOT/outputs/a800_session_monitor.log
mkdir -p "$ROOT"/outputs

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

declare -A LAUNCHERS=(
  [v6_mpi_isab]="$ROOT/scripts/tmux_v6_mpi_isab.sh"
  [v6_h36m_isab]="$ROOT/scripts/tmux_v6_h36m_isab.sh"
  [v7_mixed_precision]="$ROOT/scripts/tmux_v7_mixed_precision.sh"
  [v8_mixed_robust]="$ROOT/scripts/tmux_v8_mixed_robust.sh"
)

log "Monitor started (PID $$)."
while true; do
  for session in "${!LAUNCHERS[@]}"; do
    launcher="${LAUNCHERS[$session]}"
    python_alive=$(pgrep -f "\.venv/bin/python.*train_omniview_fusion.*${session}" >/dev/null && echo yes || echo no)
    tmux_alive=$(tmux has-session -t "$session" 2>/dev/null && echo yes || echo no)
    if [[ "$python_alive" == "no" && "$tmux_alive" == "no" ]]; then
      log "WARNING: $session missing (no python, no tmux). Restarting via $launcher"
      if [[ -x "$launcher" ]]; then
        bash "$launcher" || log "ERROR: failed to restart $session"
      else
        log "ERROR: launcher not executable: $launcher"
      fi
    fi
  done
  sleep $INTERVAL
done
