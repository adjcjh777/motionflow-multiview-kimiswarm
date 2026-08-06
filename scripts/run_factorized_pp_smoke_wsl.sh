#!/usr/bin/env bash
# Smoke-test the factorised ST+PP model on MPI-INF-3DHP.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --model_type factorized --n_view_layers 2 --n_temporal_layers 2 \
  --residual_hidden 64 --epochs 5 --train_samples 500 --batch_size 8 --val_stride 50 \
  --pp_loss_weight 0.1 --cam_aug_pp 3.0 --cam_aug_focal 0.01 \
  --output outputs/factorized_pp_smoke_mpiinf3dhp.pth \
  "$@" \
  > outputs/factorized_pp_smoke_mpiinf3dhp.log 2>&1
