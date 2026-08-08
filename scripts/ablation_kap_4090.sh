#!/usr/bin/env bash
# Fast local 4090 ablation: v18 (no KAP) vs v23 (with KAP).
# Runs one epoch with a tiny model to get a quick val_MPJPE comparison.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
PYTHON=${PYTHON:-.venv/bin/python3}

# Shared tiny config.
BASE_FLAGS="
  --use_mixed_loader
  --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml
  --use_full_precision_dlt
  --use_robust_dlt_reweight
  --use_irls_reweight
  --use_domain_embedding
  --use_deformable_cross_view_attention_v18
  --num_workers 0
  --d 64 --residual_hidden 128 --n_st_layers 2
  --graph_num_layers 1 --n_joint_layers 1 --n_heads 4
  --epochs 1 --batch_size 32 --train_samples 50 --val_stride 10
  --lr 1e-3 --lr_cosine --lr_warmup_epochs 0 --lr_min 1e-6
  --max_grad_norm 1.0 --ema_decay 0.999
  --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true
  --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true
  --use_entropy_regularization true --attention_entropy_weight 0.01
  --use_camera_view_embedding --use_set_view_aggregator
  --use_variable_view_training
  --variable_view_min_views 2 --variable_view_max_views 4
  --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0
  --variable_view_permute
  --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0
  --reproj_loss_weight 0.1 --reproj_warmup_epochs 0
  --aleatoric_reproj_loss_weight 0.1
  --outlier_view_prob 0.0
"

echo "=== v18 (no KAP) ==="
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py $BASE_FLAGS \
    --output outputs/ablation_v18_no_kap.pth \
    > outputs/ablation_v18_no_kap.log 2>&1
VAL_V18=$(grep 'val_MPJPE=' outputs/ablation_v18_no_kap.log | tail -n 1 || true)
echo "$VAL_V18"

echo "=== v23 (KAP) ==="
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py $BASE_FLAGS \
    --use_kinematic_anthropometric_prior_v22 --kap_loss_weight 0.01 \
    --output outputs/ablation_v23_kap.pth \
    > outputs/ablation_v23_kap.log 2>&1
VAL_V23=$(grep 'val_MPJPE=' outputs/ablation_v23_kap.log | tail -n 1 || true)
echo "$VAL_V23"

echo "=== Summary ==="
echo "v18: $VAL_V18"
echo "v23: $VAL_V23"
