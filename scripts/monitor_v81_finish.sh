#!/usr/bin/env bash
set -euo pipefail
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

LOG="outputs/ablations/v81_true_gt_h36m_medium_a800.log"
DONE="outputs/ablations/v81_true_gt_h36m_medium_a800.eval_done"
EVAL_LOG="outputs/eval_v81_true_gt_h36m_test_a800.log"

# Wait for the v81 training processes to finish
while pgrep -af "train_omniview_fusion_v5_webbridge_multi.py" | grep -q "v81_true_gt_h36m_medium_a800"; do
    sleep 30
done

# Give the log a moment to flush
sleep 10

echo "v81 training finished. Running S9/S11 test evaluation..."

# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed on A800.
# Default to GPU 6; override with CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python -u scripts/eval_v81_true_gt_h36m_test.py \
    --checkpoint outputs/ablations/v81_true_gt_h36m_medium_a800.pth \
    --config_json outputs/ablations/v81_true_gt_h36m_medium_a800.config.json \
    --val_stride 13 \
    --batch_size 8 \
    --out_json outputs/eval_v81_true_gt_h36m_test_a800.json \
    > "${EVAL_LOG}" 2>&1

touch "${DONE}"
echo "eval complete" >> "${DONE}"
