#!/usr/bin/env bash
# Launch v23b (KAP 0.001, no BA) on GPU4 and v24b (fixed BA + KAP 0.001) on GPU6.
set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20
VENV="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u"

run() {
    gpu=$1
    name=$2
    script=$3
    tmux kill-session -t "$name" 2>/dev/null || true
    tmux new-session -d -s "$name" \
        "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/mnt/nvme0n1/zhangzy/motionflow-multiview-kimiswarm-iter20 $VENV bash $script"
}

run 4 v23b_kap001_no_ba_gpu4 scripts/run_v23b_kap001_no_ba_a800_small.sh
run 6 v24b_kap001_fixed_ba_gpu6 scripts/run_v24b_kap001_fixed_ba_a800_small.sh

sleep 2
tmux list-sessions
