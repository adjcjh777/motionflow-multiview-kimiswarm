#!/usr/bin/env bash
# Full principal-point correction model on WSL + RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

TRAIN=(
  data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz
)
VAL=data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz

python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --num_workers 0 \
  --train "${TRAIN[@]}" --val "$VAL" \
  --clip_len 13 --d 64 --residual_hidden 128 --principal_point_hidden 64 \
  --epochs 5 --train_samples 1000 --val_stride 50 --batch_size 8 \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0 \
  --pp_loss_weight 0.1 \
  --output outputs/principal_point_pp_supervised_full.pth \
  "$@"
