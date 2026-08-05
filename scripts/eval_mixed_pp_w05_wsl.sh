#!/usr/bin/env bash
# Evaluate mixed-dataset PP small model (pp_loss_weight=0.05) on MPI and H36M.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CHECKPOINT=outputs/mixed_pp_small_w05.pth

python -u experiments/eval_mixed_dataset_principal_point.py \
  --checkpoint "$CHECKPOINT" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --val_dataset mpi \
  --clip_len 13 --d 32 --n_temporal_layers 2 --residual_hidden 64 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 50 \
  --out_json outputs/mixed_pp_small_w05_mpi.json

python -u experiments/eval_mixed_dataset_principal_point.py \
  --checkpoint "$CHECKPOINT" \
  --dataset data/webbridge/h36m_meters/s_01_acts_07_multiview_m.npz \
  --val_dataset h36m \
  --clip_len 13 --d 32 --n_temporal_layers 2 --residual_hidden 64 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 50 \
  --out_json outputs/mixed_pp_small_w05_h36m.json
