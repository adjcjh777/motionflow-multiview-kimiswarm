#!/usr/bin/env bash
# A800 GPU 6 launch script for Iskakov ICCV 2019 learnable triangulation
# baseline on the full AIST++ train/val split.
#
# MotionFlow-MultiView GPU policy: only GPUs 6 and 7 are allowed.
# GPU 7 is usually busy with v85; this script targets GPU 6.
#
# Launch with:
#   nohup bash scripts/run_iskakov_aistpp_full_a800_gpu6.sh &

set -euo pipefail

cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

export CUDA_VISIBLE_DEVICES=6

PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/baselines

nohup "$PYTHON" -u experiments/train_iskakov_aistpp_full.py \
    --device cuda \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_dim 32 \
    --train_samples_per_epoch 4096 \
    --patience 3 \
    --seed 20260812 \
    --log_path outputs/baselines/iskakov_learnable_tri_aistpp_full_a800_gpu6.log \
    --ckpt_path outputs/baselines/iskakov_learnable_tri_aistpp_full_a800_gpu6.pth \
    > outputs/baselines/iskakov_learnable_tri_aistpp_full_a800_gpu6_nohup.log 2>&1 &

PID=$!
echo "Launched Iskakov AIST++ full baseline on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: outputs/baselines/iskakov_learnable_tri_aistpp_full_a800_gpu6.log"
