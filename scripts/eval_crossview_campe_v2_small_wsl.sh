#!/usr/bin/env bash
# Evaluate cross-view CamPE v2 small model on MPI-INF-3DHP clean + robustness.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/eval_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
  --model_class crossview_residual_campe_v2 \
  --checkpoint outputs/ray_attention_temporal_crossview_residual_campe_v2_small.pth \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --clip_len 13 --d 32 --residual_hidden 64 --n_st_layers 2 --batch_size 8 --val_stride 50 \
  --out_json outputs/crossview_campe_v2_small_eval.json
