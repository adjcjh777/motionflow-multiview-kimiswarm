#!/usr/bin/env bash
# Variable-view inference with DLT fallback for the v86 unified
# sparse-cross-domain A800 medium checkpoint. Falls back to direct
# confidence-weighted DLT whenever n_active < n_views_max.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

CKPT="outputs/ablations/v86_sparse_cross_domain_medium_a800.pth"
CONFIG="outputs/ablations/v86_sparse_cross_domain_medium_a800.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"
OUT_PREFIX="outputs/variable_view_fix/variable_view_v86_sparse_cross_domain_medium_a800_dlt_fallback"

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

mkdir -p "outputs/variable_view_fix"

PYTHONUNBUFFERED=1 /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --dataset_manifest "$MANIFEST" \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --var_view_dlt_fallback \
    --output_csv "${OUT_PREFIX}.csv" \
    --output_json "${OUT_PREFIX}.json" \
    > "${OUT_PREFIX}.log" 2>&1

echo "v86 A800 sparse-cross-domain variable-view DLT-fallback eval complete. Results: ${OUT_PREFIX}.{csv,json}"
