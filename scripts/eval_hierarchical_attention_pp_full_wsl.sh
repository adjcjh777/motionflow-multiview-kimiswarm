#!/usr/bin/env bash
# Full clean-metrics evaluation for the Hierarchical Attention full checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MF_VENV:-/tmp/mf_venv}
. "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

LOG="outputs/eval_hierarchical_attention_pp_full_mpiinf3dhp.log"
mkdir -p "$(dirname "$LOG")"

python -u experiments/eval_full_metrics.py \
    --model hierarchical_view_temporal_joint_pp \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/hierarchical_attention_pp_full_mpiinf3dhp.pth \
    --clip_len 13 --d 64 --residual_hidden 128 \
    --n_view_groups 2 --n_view_layers 2 --n_temporal_layers 2 --n_joint_graph_layers 1 \
    --val_stride 50 \
    --output_json outputs/hierarchical_attention_pp_full_mpiinf3dhp_eval.json \
    > "$LOG" 2>&1
