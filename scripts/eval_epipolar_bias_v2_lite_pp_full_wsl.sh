#!/usr/bin/env bash
# Full-metrics evaluation for a trained Epipolar Bias v2 Lite checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python experiments/eval_full_metrics.py \
  --model epipolar_bias_v2_lite_pp \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --checkpoint outputs/epipolar_bias_v2_lite_pp_full_mpiinf3dhp.pth \
  --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
  --val_stride 50 \
  --output_json outputs/epipolar_bias_v2_lite_pp_full_eval.json
