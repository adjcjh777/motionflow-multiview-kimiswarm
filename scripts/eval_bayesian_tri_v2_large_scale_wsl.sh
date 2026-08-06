#!/usr/bin/env bash
# Full clean-metrics evaluation for the large-scale Bayesian triangulation v2 checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/eval_full_metrics.py \
    --model bayesian_tri_v2_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth \
    --clip_len 13 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --val_stride 50 \
    --output_json outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_eval.json
