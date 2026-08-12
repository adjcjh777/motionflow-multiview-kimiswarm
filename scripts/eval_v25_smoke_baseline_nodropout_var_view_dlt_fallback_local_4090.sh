#!/usr/bin/env bash
# Variable-view evaluation (k=2/3/4, DLT fallback) for the no-dropout smoke checkpoint.
set -euo pipefail

PYTHON=${PYTHON:-python}

CKPT="outputs/v25_true_gt_h36m_smoke_baseline_nodropout_local_4090.pth"
CONFIG="outputs/v25_true_gt_h36m_smoke_baseline_nodropout_local_4090.config.json"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    exit 1
fi

$PYTHON -u experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --dataset_manifest tmp/h36m_true_gt_val_manifest_subsampled.txt \
    --clip_len 9 \
    --k_values 2 3 4 \
    --num_subsets_per_k 5 \
    --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_v25_smoke_baseline_nodropout_local_4090_dlt_fallback.csv \
    --output_json outputs/variable_view_v25_smoke_baseline_nodropout_local_4090_dlt_fallback.json \
    > outputs/variable_view_v25_smoke_baseline_nodropout_local_4090_dlt_fallback.log 2>&1
