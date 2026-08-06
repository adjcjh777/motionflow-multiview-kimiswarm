#!/usr/bin/env bash
# Re-train cross-view PP model with stronger geometry supervision to fix the
# principal-point correction head (see issue #21 / PR #17).
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
  --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 20 \
  --train_samples 1000 --batch_size 8 --val_stride 50 \
  --pp_loss_weight 0.5 --reproj_weight 0.2 --cam_aug_pp 5.0 --cam_aug_focal 0.01 \
  --cam_aug_schedule intrinsics_curriculum --cam_aug_intrinsics_ramp_epochs 5 \
  --pp_pretrain_epochs 3 \
  --output outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth \
  "$@"
