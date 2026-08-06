#!/usr/bin/env bash
# Train temporal consistency model: longer clips (clip_len=25) + velocity loss.
# GPU-only; run only when the RTX 4090 queue is free.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 25 --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 10 \
  --train_samples 1000 --batch_size 4 --val_stride 50 \
  --pp_loss_weight 0.1 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --velocity_loss_weight 0.05 \
  --warm_start outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
  --output outputs/ray_attention_temporal_crossview_residual_principal_point_velocity_longclip.pth \
  "$@"
