#!/usr/bin/env bash
# Evaluate the full pp-supervised checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"

python -u experiments/eval_principal_point_model_mpiinf3dhp.py \
  --checkpoint outputs/principal_point_pp_supervised_full.pth \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 64 --residual_hidden 128 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 10 \
  --out_json outputs/principal_point_pp_supervised_full_eval.json
