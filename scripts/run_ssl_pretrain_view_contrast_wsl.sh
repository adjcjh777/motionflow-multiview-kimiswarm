#!/usr/bin/env bash
# Self-supervised masked-view pretraining WITH cross-view contrastive loss on WSL RTX 4090.
# This variant adds --lambda_contrast to the standard SSL pretext task, encouraging
# per-joint features to be invariant across camera views.
set -e
cd /mnt/d/WSL_workspace/about_eassys/motionflow-multivie-kimiswarm

TRAIN_FILES=(
  data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz
)
VAL_FILE=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

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
  --lambda_contrast 0.1 \
  --contrastive_dim 64 \
  --contrastive_temperature 0.07 \
  --output outputs/ray_attention_ssl_view_contrast_mpi.pth \
  "$@"
