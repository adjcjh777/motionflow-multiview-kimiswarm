#!/usr/bin/env bash
# Evaluate v25 true-GT v2 medium checkpoint on H36M test subjects (S9/S11).
#
# Usage (on A800):
#   bash scripts/run_v25_true_gt_v2_test_a800.sh
#
#   # Specify GPU (project policy: only GPU 6 or 7)
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v25_true_gt_v2_test_a800.sh

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
PYTHON="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"

# GPU discipline: default to GPU 6, allow CUDA_VISIBLE_DEVICES override.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p outputs

"${PYTHON}" -u scripts/eval_v25_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v25_true_gt_v2_medium_a800.pth \
    --config_json outputs/ablations/v25_true_gt_v2_medium_a800.config.json \
    --out_json outputs/eval_v25_true_gt_v2_h36m_test.json \
    --batch_size 8 \
    --val_stride 13 \
    > outputs/eval_v25_true_gt_v2_h36m_test.log 2>&1
