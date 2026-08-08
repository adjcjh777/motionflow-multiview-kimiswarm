#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_temporal_perceiver_v19 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --output outputs/omniview_fusion_v19_temporal_perceiver_smoke.pth \
    > outputs/omniview_fusion_v19_temporal_perceiver_smoke.log 2>&1
echo "v19 smoke complete"
