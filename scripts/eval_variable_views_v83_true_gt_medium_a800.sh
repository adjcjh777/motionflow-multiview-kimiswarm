#!/usr/bin/env bash
# Variable-view inference benchmark for v83 A800 medium checkpoint.
set -euo pipefail
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

CKPT="outputs/ablations/v83_true_gt_h36m_medium_a800.pth"
CONFIG="outputs/ablations/v83_true_gt_h36m_medium_a800.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"
OUT_PREFIX="outputs/variable_view_v83_true_gt_medium_a800"

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

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --dataset_manifest "$MANIFEST" \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --output_csv "${OUT_PREFIX}.csv" \
    --output_json "${OUT_PREFIX}.json" \
    > "${OUT_PREFIX}.log" 2>&1

echo "v83 A800 variable-view eval complete. Results: ${OUT_PREFIX}.{csv,json}"
