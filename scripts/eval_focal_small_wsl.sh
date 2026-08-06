#!/usr/bin/env bash
# Evaluate the small focal-aware checkpoint on MPI-INF-3DHP S2/Seq1.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"

python -u experiments/eval_principal_point_model_mpiinf3dhp.py \
  --checkpoint outputs/principal_point_pp_supervised_focal_small.pth \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --principal_point_hidden 64 \
  --batch_size 8 --val_stride 10 --focal_max_scale 0.1 \
  --out_json outputs/principal_point_pp_supervised_focal_small_eval.json
