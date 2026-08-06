#!/usr/bin/env bash
# Evaluate the mixed-dataset principal-point correction checkpoint on MPI-INF-3DHP S2/Seq1.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"

python -u experiments/eval_mixed_dataset_principal_point.py \
  --checkpoint outputs/mixed_pp_small.pth \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --val_dataset mpi \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 10 \
  --out_json outputs/mixed_pp_small_eval.json
