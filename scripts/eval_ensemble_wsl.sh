#!/usr/bin/env bash
# Ensemble evaluation of multiple checkpoints on MPI-INF-3DHP validation.
# Usage: scripts/eval_ensemble_wsl.sh <model_type> <output_json>
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-$(pwd)/.venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

MODEL="${1:-bayesian_tri_v2_pp}"
OUT_JSON="${2:-outputs/bayesian_tri_v2_ensemble_eval.json}"

python -u experiments/prototypes/eval_ensemble_checkpoints.py \
  --model "$MODEL" \
  --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
  --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth \
  --checkpoint outputs/bayesian_tri_v2_full_mpiinf3dhp.pth \
  --checkpoint outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth \
  --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
  --batch_size 4 --val_stride 50 \
  --output_json "$OUT_JSON"
