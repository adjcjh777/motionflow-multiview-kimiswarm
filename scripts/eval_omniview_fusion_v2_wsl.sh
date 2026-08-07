#!/usr/bin/env bash
# Evaluate a trained OmniMultiViewFusionV2 checkpoint on MPI-INF-3DHP S2/Seq1.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-.venv}
# shellcheck source=/dev/null
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

CHECKPOINT=${1:-outputs/omniview_fusion_v2_mpiinf3dhp.pth}
DATASET=${2:-data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz}

python -u experiments/eval_omniview_fusion_v2_mpiinf3dhp.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --clip_len 13 \
    --d 128 \
    --residual_hidden 128 \
    --n_st_layers 2 \
    --graph_num_layers 1 \
    --n_heads 4 \
    --batch_size 8 \
    --val_stride 50 \
    --run_robustness \
    --run_variable_views \
    --out_json outputs/eval_omniview_fusion_v2_mpiinf3dhp.json \
    --out_csv outputs/eval_omniview_fusion_v2_mpiinf3dhp.csv \
    "$@"
