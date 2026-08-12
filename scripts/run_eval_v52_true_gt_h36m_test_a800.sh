#!/usr/bin/env bash
# Run v52 UWT H36M true-GT test evaluation on A800-D.
#
# Uses the best checkpoint from the v52 UWT true-GT run:
#   outputs/ablations/v52_true_gt_h36m_a800.pth
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
# Default to GPU 6; override with CUDA_VISIBLE_DEVICES.

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

echo "Evaluating v52 UWT true-GT checkpoint on H36M test S9/S11 (GPU ${CUDA_VISIBLE_DEVICES})"

$PYTHON -u scripts/eval_v52_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v52_true_gt_h36m_a800.pth \
    --config_json outputs/ablations/v52_true_gt_h36m_a800.config.json \
    --val_stride 13 \
    --batch_size 8 \
    --out_json outputs/eval_v52_true_gt_h36m_test_a800.json

echo "Done. Results: outputs/eval_v52_true_gt_h36m_test_a800.json"
