#!/usr/bin/env bash
# Launch v25 + KAP on A800-D in a persistent tmux session.
set -euo pipefail

GPU=${1:-7}
MODE=${2:-small}  # small or full
NAME="v25_kap001_${MODE}_gpu${GPU}"

PYTHONPATH="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
PYTHON_BIN="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

if [ "$MODE" = "small" ]; then
    SCRIPT="scripts/run_v25_kap001_a800_small.sh"
elif [ "$MODE" = "full" ]; then
    SCRIPT="scripts/run_v25_kap001_a800_full.sh"
else
    echo "Usage: $0 <gpu> <small|full>"
    exit 1
fi

tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new-session -d -s "$NAME" \
    "export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PYTHONPATH; export PYTHON=$PYTHON_BIN; bash $SCRIPT"

sleep 2
tmux list-sessions | grep "$NAME" || { echo "Failed to launch $NAME"; exit 1; }
