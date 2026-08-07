#!/usr/bin/env bash
# Generic 6-axis robustness matrix launcher for any registered model.
#
# Usage:
#   scripts/eval_robustness_matrix_model_wsl.sh crossview_residual_pp \
#       outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth
#
#   scripts/eval_robustness_matrix_model_wsl.sh hierarchical_view_temporal_joint_pp \
#       outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth \
#       --n_view_groups 2 --n_view_layers 2 --n_temporal_layers 2 --n_joint_graph_layers 1
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

MODEL=${1:-crossview_residual_pp}
CHECKPOINT=${2:-outputs/ray_attention_temporal_crossview_residual_principal_point_robust_retrain.pth}
shift 2 || true

python -u experiments/eval_robustness_matrix_pp_mpiinf3dhp.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --out_json "outputs/robustness_matrix_${MODEL}.json" \
    --out_md "docs/robustness_matrix_${MODEL}.md" \
    "$@"
