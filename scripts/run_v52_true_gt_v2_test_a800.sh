#!/usr/bin/env bash
# Evaluate v52 true-GT v2 medium checkpoint on H36M test subjects (S9/S11).
#
# Usage (on A800):
#   bash scripts/run_v52_true_gt_v2_test_a800.sh
#
#   # Specify GPU (project policy: only GPU 6 or 7)
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v52_true_gt_v2_test_a800.sh

set -euo pipefail

REPO="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
PYTHON="/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES

cd "${REPO}"
mkdir -p outputs

"${PYTHON}" -u scripts/eval_v52_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v52_true_gt_v2_medium_a800.pth \
    --config_json outputs/ablations/v52_true_gt_v2_medium_a800.config.json \
    --s9 data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --s11 data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
    --out_json outputs/eval_v52_true_gt_v2_h36m_test.json \
    --batch_size 8 \
    --val_stride 13 \
    > outputs/eval_v52_true_gt_v2_h36m_test.log 2>&1
