#!/usr/bin/env bash
# Full clean-metrics evaluation for the Gaussian-Splatting PP full checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/eval_full_metrics.py \
    --model crossview_residual_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/splat_pp_full_mpiinf3dhp.pth \
    --clip_len 13 \
    --d 64 \
    --residual_hidden 128 \
    --n_st_layers 2 \
    --val_stride 50 \
    --output_json outputs/splat_pp_full_mpiinf3dhp_eval.json
