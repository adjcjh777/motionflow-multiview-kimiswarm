#!/usr/bin/env bash
# Augmented large-scale MPI-INF-3DHP training for Bayesian triangulation v2.
# Adds multiview augmentation: per-view noise, joint dropout, and view dropout.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

mkdir -p outputs

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
  --model_type bayesian_tri_v2 \
  --epochs 50 --train_samples 2000 --batch_size 8 --val_stride 50 \
  --lr 3e-4 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 \
  --grad_clip_norm 1.0 \
  --pp_loss_weight 0.2 --epipolar_loss_weight 0.02 --reproj_weight 0.0 \
  --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule extended_intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --view_noise_std 1.0 --joint_dropout_rate 0.1 --view_dropout_rate 0.1 --min_views 2 \
  --pp_pretrain_epochs 3 \
  --output outputs/bayesian_tri_v2_aug_mpiinf3dhp.pth
