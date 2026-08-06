#!/usr/bin/env bash
# Full GPU training for the view-synced temporal jitter augmentation wrapper.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

LOG="outputs/multiview_sync_aug_pp_full_mpiinf3dhp.log"
mkdir -p "$(dirname "$LOG")"

python -u experiments/train_multiview_sync_aug_full_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 \
  --epochs 20 --train_samples 1000 --batch_size 8 --val_stride 50 \
  --pp_loss_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --pp_pretrain_epochs 3 \
  --output outputs/multiview_sync_aug_pp_full_mpiinf3dhp.pth \
  > "$LOG" 2>&1
