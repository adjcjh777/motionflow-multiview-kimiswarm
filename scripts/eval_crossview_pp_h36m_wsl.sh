#!/usr/bin/env bash
# Evaluate cross-view PP small model on Human3.6M.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_h36m_small.pth \
  --dataset data/webbridge/h36m_meters/s_05_acts_02_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 \
  --batch_size 8 --val_stride 50 \
  --out_json outputs/crossview_pp_h36m_small_eval.json \
  "$@"
