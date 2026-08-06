#!/usr/bin/env bash
# 6-axis robustness matrix for the PP robust re-train checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

python -u experiments/eval_robustness_matrix_pp_mpiinf3dhp.py \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --out_json outputs/robustness_matrix_pp_robust_retrain.json \
    --out_md docs/robustness_matrix_pp_robust_retrain.md
