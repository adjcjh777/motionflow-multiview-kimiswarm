#!/usr/bin/env bash
# Variable-view inference benchmark for the best cross-view PP checkpoint.
set -e
cd "$(dirname "$0")/.."

. /tmp/mf_venv/bin/activate

python experiments/eval_variable_views.py \
    --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
    --checkpoint outputs/ray_attention_temporal_crossview_residual_principal_point_full_ppw005_20ep.pth \
    --model_class crossview_residual_pp \
    --d 64 --residual_hidden 128 --n_temporal_layers 2 --clip_len 13 \
    --min_views 2 --max_views 14 --num_subsets_per_k 50
