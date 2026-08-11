#!/usr/bin/env bash
# A800 GPU 6 launch script for v25 H36M true-GT + AIST++ mixed-dataset training.
#
# This run applies the conservative "stability" recipe that yielded the best
# true-GT H36M result (v25_true_gt_stability_a800, 31.56 mm test weighted)
# to the mixed-dataset loader. The earlier mixed-dataset attempt
# (run_v25_ablation_mixed_dataset_a800.sh) used a more aggressive LR/warmup
# schedule and diverged at Epoch 3.
#
# Usage:
#   nohup bash scripts/run_v25_mixed_dataset_stability_a800_gpu6.sh \
#       > outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6_nohup.log 2>&1 &
#
# Requirements before launch:
#   - AIST++ canonical .npz files are present at
#     data/webbridge/aistpp_canonical/ on A800.
#   - GPU 6 is free (check nvidia-smi on a800-D).

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: pin to GPU 6 explicitly.
export CUDA_VISIBLE_DEVICES=6

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20/.venv/bin/python}

mkdir -p outputs/ablations

# v25 H36M true-GT + AIST++ mixed-dataset training with the stability recipe.
# Key differences from the original mixed-dataset run:
#   - Lower initial LR (1e-4 vs 5e-4)
#   - Longer warmup (4 epochs vs 2)
#   - Variable-view max 4 views (vs 9) and no permutation
#   - Monitors MPJPE for checkpoint selection (trainer default)
nohup "$PYTHON" -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_aist_mixed_train_val_a800.yaml \
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
    --num_workers 4 \
    --d 128 --residual_hidden 256 --n_st_layers 3 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 20 --batch_size 16 --train_samples 4096 --val_stride 20 \
    --lr 1e-4 --lr_cosine --lr_warmup_epochs 4 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --weight_decay 1e-4 \
    --early_stopping_patience 3 --early_stopping_min_delta 0.001 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.15 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.pth \
    > outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log 2>&1 &

PID=$!
echo "Launched v25 mixed-dataset stability run on GPU ${CUDA_VISIBLE_DEVICES} (PID: ${PID})"
echo "Log: outputs/ablations/v25_true_gt_mixed_dataset_stability_a800_gpu6.log"
