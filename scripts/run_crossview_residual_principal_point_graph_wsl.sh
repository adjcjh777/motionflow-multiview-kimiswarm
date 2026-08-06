#!/usr/bin/env bash
# Warm-start the skeleton-graph PP model from the current best PP checkpoint.
# This script is meant to be queued after the currently-running PP curriculum
# finishes, because it requires a GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="outputs/crossview_residual_principal_point/curriculum/best_model.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "Best PP checkpoint not found at $CKPT" >&2
    echo "Please update CKPT in this script before training." >&2
    exit 1
fi

python -m experiments.run_multiview_fusion \
    --model ray_attention_temporal_crossview_residual_principal_point_graph \
    --j 28 \
    --d 64 \
    --n_views 14 \
    --n_st_layers 2 \
    --graph_num_layers 1 \
    --warm_start "$CKPT" \
    --epochs 20 \
    --batch_size 4 \
    --lr 3e-4 \
    --train_dataset mpiinf3dhp \
    --train_split Train \
    --val_dataset mpiinf3dhp \
    --val_split Test/TS1 \
    --focal_max_scale 0.0 \
    --principal_point_max_offset 20.0 \
    --exp_name graph_joint_relation_pp_warmstart \
    "$@"
