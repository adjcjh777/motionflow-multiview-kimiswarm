#!/usr/bin/env bash
# Evaluate the visibility + uncertainty v1 checkpoint on MPI-INF-3DHP.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CKPT="outputs/crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.pth"
DATA="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

python experiments/eval_crossview_residual_visibility_uncertainty_v1.py     --dataset "$DATA"     --checkpoint "$CKPT"     --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128     --batch_size 8     --output_json "outputs/eval_crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.json"
