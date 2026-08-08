#!/usr/bin/env bash
# CPU smoke test for v25 + v18 top-k ST combination.
set -euo pipefail

CUDA_VISIBLE_DEVICES=9 python experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --smoke \
    --use_deformable_cross_view_attention_v18 --deformable_attention_use_topk_st \
    --use_multiview_geometry_fusion_v25 \
    --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment \
    --epochs 1 --batch_size 2 --train_samples 10 --val_stride 2 \
    --output outputs/v25_v18_topk_st_cpu_smoke.pth \
    > outputs/v25_v18_topk_st_cpu_smoke.log 2>&1

echo "v25 + v18 top-k ST CPU smoke complete"
