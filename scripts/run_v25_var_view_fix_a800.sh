#!/usr/bin/env bash
# Re-run v25 stability variable-view evaluation on A800
# using the fixed variable_view_inference wrapper.
#
# Usage:
#   nohup bash scripts/run_v25_var_view_fix_a800.sh &

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
# Default to GPU 6; override with CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/variable_view_fix

nohup "$PYTHON" -u experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v25_true_gt_stability_a800.pth \
    --config outputs/ablations/v25_true_gt_stability_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_val_manifest.txt \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 20 \
    --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json \
    > outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback_nohup.log 2>&1 &

PID=$!
echo "Launched v25 var-view re-eval on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_fixed_nohup.log"
