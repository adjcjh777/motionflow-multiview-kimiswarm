#!/usr/bin/env bash
# Variable-view evaluation of the v81 temporal-pose-attention checkpoint with
# direct DLT fallback for sparse views (k<4).
#
# Usage:
#   # Default GPU 6 (override with CUDA_VISIBLE_DEVICES if busy)
#   bash scripts/run_v81_var_view_dlt_fallback_a800_gpuX.sh
#
#   # Run on a specific free GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v81_var_view_dlt_fallback_a800_gpuX.sh

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/variable_view_fix

nohup "$PYTHON" -u experiments/eval_variable_views.py \
    --model_class omniview_v5 \
    --checkpoint outputs/ablations/v81_true_gt_h36m_medium_a800.pth \
    --config outputs/ablations/v81_true_gt_h36m_medium_a800.config.json \
    --dataset_manifest tmp/h36m_true_gt_val_manifest.txt \
    --clip_len 13 \
    --min_views 2 \
    --max_views 4 \
    --num_subsets_per_k 50 \
    --seed 42 \
    --var_view_dlt_fallback \
    --output_csv outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback.csv \
    --output_json outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback.json \
    > outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_nohup.log 2>&1 &

PID=$!
echo "Launched v81 var-view DLT-fallback eval on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: outputs/variable_view_fix/variable_view_v81_true_gt_medium_a800_dlt_fallback_nohup.log"
