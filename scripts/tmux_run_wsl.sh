#!/usr/bin/env bash
# Run a MotionFlow-MultiView training script inside a persistent tmux session.
# Usage: scripts/tmux_run_wsl.sh <path-to-run-script> [session-name]
set -euo pipefail

SCRIPT_PATH="${1:?Usage: scripts/tmux_run_wsl.sh <run-script.sh> [session-name]}"
SCRIPT_NAME="$(basename "$SCRIPT_PATH" .sh)"
SESSION_NAME="${2:-${SCRIPT_NAME}}"

LOG_DIR="outputs/tmux_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach: tmux attach -t $SESSION_NAME"
    echo "Tail log: tail -f $LOG_FILE"
    exit 0
fi

echo "Starting tmux session '$SESSION_NAME' for $SCRIPT_PATH"
echo "Log file: $LOG_FILE"

cd "$(dirname "$0")/.."

tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'echo Running $SCRIPT_PATH; exec bash $SCRIPT_PATH' 2>&1 | tee -a $LOG_FILE"

echo "Session '$SESSION_NAME' started."
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Detach (inside tmux): Ctrl+b then d"
echo "Tail log: tail -f $LOG_FILE"
