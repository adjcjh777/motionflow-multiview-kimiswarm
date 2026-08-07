#!/usr/bin/env bash
# Improved A800 session monitor for the v10-v16 omniview experiments.
# Usage:
#   scripts/a800_session_monitor.sh [--dry-run] [--once] [--start-missing]
#
# --dry-run : print actions without killing or starting tmux sessions.
# --once    : run a single pass and exit (useful for tests/cron).
# --start-missing : also start sessions that have never been seen before.

set -uo pipefail

ROOT=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
INTERVAL=60
LOG=${ROOT}/outputs/a800_session_monitor.log
STATE=${ROOT}/outputs/a800_session_monitor.seen
DRY_RUN=0
ONCE=0
AUTO_START_NEW=0

usage() {
  echo "Usage: $0 [--dry-run] [--once] [--start-missing]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    --once)           ONCE=1; shift ;;
    --start-missing)  AUTO_START_NEW=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$LOG")"
touch "$LOG" "$STATE"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# Experiment registry: session name -> GPU, launcher, unique process marker.
# ---------------------------------------------------------------------------
SESSIONS=(
  v10_aleatoric_outlier_a800
  v10_no_outlier
  v11_irls
  v12_adaptive_multiscale
  v13_temporal_a800
  v15_kinematic_a800
  v16_occlusion_a800
)

declare -A GPU
GPU[v10_aleatoric_outlier_a800]=4
GPU[v10_no_outlier]=7
GPU[v11_irls]=5
GPU[v12_adaptive_multiscale]=6
GPU[v13_temporal_a800]=0
GPU[v15_kinematic_a800]=1
GPU[v16_occlusion_a800]=2

declare -A RUN_SCRIPT
RUN_SCRIPT[v10_aleatoric_outlier_a800]="${ROOT}/scripts/run_v10_aleatoric_outlier.sh"
RUN_SCRIPT[v10_no_outlier]="${ROOT}/scripts/run_v10_no_outlier.sh"
RUN_SCRIPT[v11_irls]="${ROOT}/scripts/run_v11_irls.sh"
RUN_SCRIPT[v12_adaptive_multiscale]="${ROOT}/scripts/run_v12_adaptive_multiscale.sh"
RUN_SCRIPT[v13_temporal_a800]="${ROOT}/scripts/run_v13_temporal.sh"
RUN_SCRIPT[v15_kinematic_a800]="${ROOT}/scripts/run_v15_kinematic_chain.sh"
RUN_SCRIPT[v16_occlusion_a800]="${ROOT}/scripts/run_v16_occlusion_noise.sh"

declare -A MARKER
MARKER[v10_aleatoric_outlier_a800]="omniview_fusion_v10_aleatoric_outlier.pth"
MARKER[v10_no_outlier]="omniview_fusion_v10_no_outlier.pth"
MARKER[v11_irls]="omniview_fusion_v11_irls.pth"
MARKER[v12_adaptive_multiscale]="omniview_fusion_v12_adaptive_multiscale.pth"
MARKER[v13_temporal_a800]="omniview_fusion_v13_temporal.pth"
MARKER[v15_kinematic_a800]="omniview_fusion_v15_kinematic_chain.pth"
MARKER[v16_occlusion_a800]="omniview_fusion_v16_occlusion_noise.pth"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
mark_seen() {
  if ! grep -qx "$1" "$STATE" 2>/dev/null; then
    echo "$1" >> "$STATE"
  fi
}

is_seen() {
  grep -qx "$1" "$STATE" 2>/dev/null
}

tmux_alive() {
  tmux has-session -t "$1" >/dev/null 2>&1
}

python_alive() {
  local marker=$1
  # The python command line contains a unique --output marker for each experiment.
  pgrep -af "train_omniview_fusion_v5_webbridge_multi.py" 2>/dev/null | grep -q "$marker"
}

start_session() {
  local session=$1
  local gpu=${GPU[$session]}
  local script=${RUN_SCRIPT[$session]}
  log "START: creating tmux session $session on GPU $gpu using $(basename "$script")"
  if [[ $DRY_RUN -eq 1 ]]; then
    log "[DRY-RUN] tmux new-session -d -s $session -n main \"cd $ROOT && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=$gpu && bash $script\""
    return
  fi
  tmux new-session -d -s "$session" -n main \
    "cd $ROOT && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=$gpu && bash $script"
}

restart_session() {
  local session=$1
  local reason=${2:-crashed}
  local gpu=${GPU[$session]}
  log "RESTART: $session ($reason) on GPU $gpu"
  if tmux_alive "$session"; then
    if [[ $DRY_RUN -eq 1 ]]; then
      log "[DRY-RUN] tmux kill-session -t $session"
    else
      tmux kill-session -t "$session"
    fi
  fi
  start_session "$session"
}

check_sessions() {
  local session gpu marker t p
  for session in "${SESSIONS[@]}"; do
    gpu=${GPU[$session]}
    marker=${MARKER[$session]}

    t=$(tmux_alive "$session" && echo yes || echo no)
    p=$(python_alive "$marker" && echo yes || echo no)

    if [[ "$t" == yes && "$p" == yes ]]; then
      log "OK: $session (GPU $gpu, tmux+python alive)"
      mark_seen "$session"
    elif [[ "$t" == yes && "$p" == no ]]; then
      log "CRASHED: $session python process missing, tmux session exists"
      restart_session "$session" "python missing"
      mark_seen "$session"
    elif [[ "$t" == no && "$p" == yes ]]; then
      log "ORPHAN: $session python running without tmux (GPU $gpu)"
    else
      if is_seen "$session" || [[ $AUTO_START_NEW -eq 1 ]]; then
        log "MISSING: $session not running (GPU $gpu); starting"
        start_session "$session"
        mark_seen "$session"
      else
        log "NOT_STARTED: $session not yet launched; skipping (use --start-missing)"
      fi
    fi
  done
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
log "Monitor started. dry_run=$DRY_RUN once=$ONCE auto_start_new=$AUTO_START_NEW"

if [[ $ONCE -eq 1 ]]; then
  check_sessions
  log "Monitor run-once complete."
else
  while true; do
    check_sessions
    log "Sleeping ${INTERVAL}s..."
    sleep $INTERVAL
  done
fi
