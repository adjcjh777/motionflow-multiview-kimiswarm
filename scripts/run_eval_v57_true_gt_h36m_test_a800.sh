#!/usr/bin/env bash
# Run v57 H36M true-GT test evaluation on A800-D.
#
# Uses the best checkpoint from the v57 true-GT re-run:
#   outputs/ablations/v57_true_gt_medium_a800.pth
#
# GPU 7 is the least-utilised A800 GPU at the time of writing, but the run is
# short and lightweight enough to be moved to any free GPU by setting
# CUDA_VISIBLE_DEVICES accordingly.

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

echo "Evaluating v57 true-GT checkpoint on H36M test S9/S11 (GPU ${CUDA_VISIBLE_DEVICES})"

$PYTHON -u scripts/eval_v57_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v57_true_gt_medium_a800.pth \
    --config_json outputs/ablations/v57_true_gt_medium_a800.config.json \
    --val_stride 13 \
    --batch_size 8 \
    --out_json outputs/eval_v57_true_gt_h36m_test_a800.json

echo "Done. Results: outputs/eval_v57_true_gt_h36m_test_a800.json"
