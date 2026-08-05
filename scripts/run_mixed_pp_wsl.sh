#!/usr/bin/env bash
# Mixed-dataset principal-point correction training on WSL + RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

MPI_TRAIN=(
  data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz
  data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz
)
H36M_TRAIN=(
  data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz
  data/webbridge/h36m_meters/s_01_acts_03_multiview_m.npz
  data/webbridge/h36m_meters/s_01_acts_04_multiview_m.npz
  data/webbridge/h36m_meters/s_01_acts_05_multiview_m.npz
  data/webbridge/h36m_meters/s_01_acts_06_multiview_m.npz
)

python -u experiments/train_mixed_dataset_principal_point.py \
  --mpi_train "${MPI_TRAIN[@]}" \
  --h36m_train "${H36M_TRAIN[@]}" \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --val_dataset mpi \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --epochs 10 --train_samples 500 --val_stride 50 --batch_size 8 --num_workers 0 \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0 \
  --pp_loss_weight 0.1 \
  --output outputs/mixed_pp_small.pth \
  "$@"
