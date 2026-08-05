#!/usr/bin/env bash
# Train cross-view CamPE v2 small model on MPI-INF-3DHP (WSL + RTX 4090).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/train_ray_attention_temporal_crossview_residual_mpiinf3dhp.py \
  --model_class crossview_residual_campe_v2 \
  --train \
    data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
    data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
    data/webbridge/mpi_inf_3dhp/s_03_seq_01_v14_multiview_m.npz \
    data/webbridge/mpi_inf_3dhp/s_03_seq_02_v14_multiview_m.npz \
  --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --epochs 10 --train_samples 500 --batch_size 8 \
  --output outputs/ray_attention_temporal_crossview_residual_campe_v2_small.pth \
  "$@"
