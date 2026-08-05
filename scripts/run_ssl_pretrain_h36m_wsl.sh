#!/usr/bin/env bash
# Self-supervised masked-view pretraining on H36M (WSL RTX 4090).
set -e
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm

# Subjects 1,5,6,7,8 for training; subject 9 for validation.
TRAIN_FILES=(
  data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_05_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_06_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_07_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_08_acts_02_multiview_m.npz
)
VAL_FILE=data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz

conda run -n mf python experiments/pretrain_ray_attention_ssl.py \
  --train "${TRAIN_FILES[@]}" \
  --val "${VAL_FILE}" \
  --clip_len 13 \
  --d 64 \
  --residual_hidden 128 \
  --n_st_layers 2 \
  --epochs 50 \
  --batch_size 8 \
  --train_samples 4000 \
  --mask_ratio 0.25 \
  --mask_mode mixed \
  --lambda_vis 1.0 \
  --lambda_mask 1.0 \
  --lambda_smooth 0.1 \
  --lambda_bone 0.1 \
  --output outputs/ray_attention_ssl_h36m.pth \
  "$@"
