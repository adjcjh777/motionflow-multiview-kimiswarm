#!/usr/bin/env bash
# Fast small ablation of explicit principal-point supervision on WSL + RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the native-WSL GPU venv (avoids slow /mnt/d package installs)
VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
         data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 --epochs 10 --num_workers 4 \
  --train_samples 500 --val_stride 50 --batch_size 8 --pp_loss_weight 0.1 \
  --cam_aug_rot 0.5 --cam_aug_trans 0.005 --cam_aug_focal 0.01 --cam_aug_pp 5.0 \
  --output outputs/principal_point_pp_supervised_small.pth \
  "$@"
