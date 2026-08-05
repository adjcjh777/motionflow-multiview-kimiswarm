#!/usr/bin/env bash
# Self-supervised masked-view pre-training on H36M (full train/val split).
set -e
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate

TRAIN_DIR="data/webbridge/h36m"

echo "Train subjects: 1,5,6,7,8; Val: 9; Test: 11"

python experiments/pretrain_ray_attention_ssl.py \
    --train \
        ${TRAIN_DIR}/s_01_acts_*_multiview.npz \
        ${TRAIN_DIR}/s_05_acts_*_multiview.npz \
        ${TRAIN_DIR}/s_06_acts_*_multiview.npz \
        ${TRAIN_DIR}/s_07_acts_*_multiview.npz \
        ${TRAIN_DIR}/s_08_acts_*_multiview.npz \
    --val \
        ${TRAIN_DIR}/s_09_acts_02_multiview.npz \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --epochs 30 --batch_size 16 --train_samples 4000 --val_stride 10 \
    --mask_ratio 0.25 --mask_mode mixed \
    --lambda_vis 1.0 --lambda_mask 1.0 --lambda_smooth 0.1 --lambda_bone 0.1 \
    --output outputs/ray_attention_ssl_h36m.pth
