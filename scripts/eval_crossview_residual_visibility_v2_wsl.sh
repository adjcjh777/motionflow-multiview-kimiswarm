#!/usr/bin/env bash
# Evaluate the visibility-gated cross-view residual v2 checkpoint on MPI-INF-3DHP.
set -euo pipefail
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate

CKPT="outputs/ray_attention_temporal_crossview_residual_visibility_v2_mpiinf3dhp.pth"
DATA="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz"

python experiments/eval_full_metrics.py \
    --model crossview_residual_pp_visibility_v2 \
    --dataset "$DATA" \
    --checkpoint "$CKPT" \
    --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
    --batch_size 8 \
    --output_json "outputs/eval_crossview_residual_visibility_v2_mpiinf3dhp.json"
