#!/usr/bin/env bash
# Launch the full-scale v26+UDP/GMM/v28 local 4090 queue inside a tmux session.
set -euo pipefail

SESSION="v26_full_queue_local_4090"
LOG="outputs/v26_full_queue_local_4090_tmux.log"

mkdir -p outputs
# Detach any existing session with the same name.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Start a new detached tmux session that runs the full queue.
# Stdout/stderr are also tee'd to a log file for easy inspection.
tmux new-session -d -s "$SESSION" \
    "bash scripts/run_v26_full_queue_local_4090.sh 2>&1 | tee -a ${LOG}"

echo "Launched full v26+UDP/GMM/v28 queue in tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "View tmux log with: tail -f $LOG"
