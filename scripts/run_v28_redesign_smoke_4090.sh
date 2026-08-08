#!/usr/bin/env bash
# v28 redesign smoke test on local RTX 4090.
# Conservative refiner: bounded residual, LayerNorm/dropout, robust floor.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
OUTPUT=${OUTPUT:-outputs/omniview_fusion_v28_redesign_smoke_4090.pth}
LOG=${LOG:-outputs/omniview_fusion_v28_redesign_smoke_4090.log}

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_physical_space_alignment_v28 \
    --v28_floor_loss_weight 0.001 \
    --v28_bone_temporal_weight 0.001 \
    --v28_residual_reg_weight 0.0001 \
    --output $OUTPUT \
    > $LOG 2>&1
