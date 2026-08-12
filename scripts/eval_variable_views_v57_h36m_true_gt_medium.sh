#!/usr/bin/env bash
# Variable-view inference benchmark for v57 H36M true-GT medium checkpoint.
# Runs on GPU; do not launch while another GPU training/eval job is active.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="outputs/omniview_fusion_v57_h36m_true_gt_medium.pth"
CONFIG="outputs/omniview_fusion_v57_h36m_true_gt_medium.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"
OUT_PREFIX="tmp/variable_view_v57_h36m_true_gt_medium"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    echo "       v57 H36M true-GT medium training is still in progress." >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Dataset manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "tmp"

# H36M true-GT has 4 cameras; evaluate variable view counts 2-4.
python experiments/eval_variable_views.py \
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
    --output_json "${OUT_PREFIX}.json"

echo "v57 variable-view eval complete. Results: ${OUT_PREFIX}.{csv,json}"
