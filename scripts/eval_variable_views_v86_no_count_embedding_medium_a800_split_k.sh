#!/usr/bin/env bash
# Split v86 no-count-embedding variable-view eval into per-k chunks to survive
# environments where long-running processes are terminated. Each chunk runs
# only one view count, so each finishes much faster than the full 2/3/4 sweep.
set -euo pipefail

# Ignore SIGTERM/SIGHUP from detached session cleanup so the eval keeps running.
trap '' TERM HUP INT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

CKPT="outputs/ablations/v86_no_count_embedding_medium_a800.pth"
CONFIG="outputs/ablations/v86_no_count_embedding_medium_a800.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"
OUT_PREFIX="outputs/variable_view_v86_no_count_embedding_medium_a800"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "outputs"

PYTHON=/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python

for K in 2 3 4; do
    echo "[$(date -Iseconds)] Starting v86 no-count-embedding eval for k=${K}"
    $PYTHON -u experiments/eval_variable_views.py \
        --model_class omniview_v5 \
        --checkpoint "$CKPT" \
        --config "$CONFIG" \
        --dataset_manifest "$MANIFEST" \
        --clip_len 13 \
        --k_values "$K" \
        --num_subsets_per_k 50 \
        --seed 42 \
        --output_csv "${OUT_PREFIX}_k${K}.csv" \
        --output_json "${OUT_PREFIX}_k${K}.json" \
        > "${OUT_PREFIX}_k${K}.log" 2>&1
    echo "[$(date -Iseconds)] Finished k=${K}"
done

echo "v86 A800 split-k no-fallback eval complete. Results: ${OUT_PREFIX}_k{2,3,4}.{csv,json}"
