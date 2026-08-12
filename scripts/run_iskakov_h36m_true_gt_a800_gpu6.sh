#!/usr/bin/env bash
# A800 GPU 6 launch script for Iskakov ICCV 2019 learnable triangulation
# baseline on H36M true-GT standard protocol (S1,5,6,7,8 -> S9/S11).
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# GPU 7 is usually busy with v85; this script targets GPU 6.
#
# Launch with:
#   nohup bash scripts/run_iskakov_h36m_true_gt_a800_gpu6.sh &

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

export CUDA_VISIBLE_DEVICES=6

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/baselines

nohup "$PYTHON" -u experiments/train_iskakov_baseline_shelf_campus.py \
    --protocol h36m \
    --epochs 10 \
    --train_samples_per_epoch 4096 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 8 \
    --ref_max_frames 2000 \
    --log_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.log \
    --ckpt_path outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.pth \
    > outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6_nohup.log 2>&1 &

PID=$!
echo "Launched Iskakov H36M true-GT baseline on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: outputs/baselines/iskakov_learnable_tri_h36m_true_gt_a800_gpu6.log"
