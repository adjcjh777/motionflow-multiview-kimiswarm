#!/usr/bin/env bash
# Local RTX 4090 cross-domain smoke for v86:
# H36M true-GT v2 + AIST++ mixed loader with the separate sparse-view head.
#
# Usage
# -----
#   bash scripts/run_v86_separate_sparse_view_head_h36m_aist_true_gt_v2_smoke_local_4090.sh
#
#   # Run on a specific GPU
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_v86_separate_sparse_view_head_h36m_aist_true_gt_v2_smoke_local_4090.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

PYTHON=${PYTHON:-python}

OUT_DIR="outputs/smoke_v86_h36m_aist_true_gt_v2"
OUT_PREFIX="${OUT_DIR}/v86_separate_sparse_view_head_h36m_aist_true_gt_v2_smoke_local_4090"
mkdir -p "${OUT_DIR}"

"$PYTHON" -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_aist_mixed_smoke.yaml \
    --num_domains 3 \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.05 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_random_view_dropout_v85 \
    --v85_dropout_prob 0.3 \
    --v85_min_views 2 \
    --v85_use_count_embedding \
    --use_v86_separate_sparse_view_head \
    --v86_ssv_head_hidden 128 \
    --v86_ssv_head_n_layers 2 \
    --v86_ssv_head_dropout 0.1 \
    --v86_ssv_head_use_count_embedding \
    --num_workers 0 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 9 \
    --epochs 2 \
    --batch_size 4 \
    --train_samples 256 \
    --val_stride 20 \
    --lr 1e-4 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 \
    --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true \
    --use_camera_conditioning true \
    --use_epipolar_bias true \
    --use_context_visibility true \
    --use_skeleton_residual true \
    --use_rotation_correction true \
    --use_entropy_regularization true \
    --attention_entropy_weight 0.01 \
    --use_camera_view_embedding \
    --use_set_view_aggregator \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output "${OUT_PREFIX}.pth" \
    > "${OUT_PREFIX}.log" 2>&1

# Extract per-domain validation MPJPE from the final checkpoint history.
"$PYTHON" - <<PY
import json
import sys
import torch

ckpt_path = "${OUT_PREFIX}_final.pth"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"Could not load final checkpoint {ckpt_path}: {exc}", file=sys.stderr)
    sys.exit(0)

history = ckpt.get("history", [])
if not history:
    print("No history found in checkpoint.")
    sys.exit(0)

print("\n=== Per-epoch validation metrics (mm) ===")
for entry in history:
    val = entry.get("val", {})
    epoch = entry.get("epoch", "?")
    overall = val.get("mpjpe")
    h36m = val.get("mpjpe_domain_0")
    aist = val.get("mpjpe_domain_2")
    overall_str = f"{overall*1000:.2f}" if overall is not None else "N/A"
    h36m_str = f"{h36m*1000:.2f}" if h36m is not None else "N/A"
    aist_str = f"{aist*1000:.2f}" if aist is not None else "N/A"
    print(f"Epoch {epoch}: overall={overall_str} mm | H36M={h36m_str} mm | AIST={aist_str} mm")
PY
