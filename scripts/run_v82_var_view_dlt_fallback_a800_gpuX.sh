#!/usr/bin/env bash
# Variable-view inference for v82 H36M true-GT medium A800 checkpoint
# with DLT fallback for sparse views (k<4).
#
# Usage:
#   nohup bash scripts/run_v82_var_view_dlt_fallback_a800_gpuX.sh &
#
# The script hard-codes GPU 6 as requested for this run;
# change CUDA_VISIBLE_DEVICES below if needed.
set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

export CUDA_VISIBLE_DEVICES=6

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

CKPT="outputs/ablations/v82_true_gt_h36m_medium_a800.pth"
CONFIG="outputs/ablations/v82_true_gt_h36m_medium_a800.config.json"
MANIFEST="tmp/h36m_true_gt_val_manifest.txt"
OUT_PREFIX="outputs/variable_view_fix/variable_view_v82_true_gt_medium_a800_dlt_fallback"

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

mkdir -p outputs/variable_view_fix

nohup "$PYTHON" -u experiments/eval_variable_views.py \
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
    > "${OUT_PREFIX}.log" 2>&1 &

PID=$!
echo "Launched v82 variable-view DLT-fallback eval on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: ${OUT_PREFIX}.log"
