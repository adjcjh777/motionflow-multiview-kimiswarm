#!/usr/bin/env bash
# A800-D medium run for v52 Uncertainty-Weighted Triangulation (UWT)
# on the corrected (non-circular) H36M true-GT v2 protocol
# (S1,5,6,7,8 train -> S9/S11 val).
#
# This is the v52-only ablation: v25 + v45 + v46 + v50 + v51 + v52. It uses
# the v57 A800 medium recipe but removes the v57 DC-PSC module, so the effect
# of the v52 learnable triangulation module can be measured in isolation.
#
# GPU policy on A800: this project only uses GPUs 6 and 7. GPUs 0-5 are reserved
# for other projects and must NOT be used. Default to GPU 6; override with
# CUDA_VISIBLE_DEVICES if needed.
#
# Usage
# -----
#   # Default: run on GPU 6
#   bash scripts/run_v52_true_gt_v2_medium_a800.sh
#
#   # Run on a specific free project GPU
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_v52_true_gt_v2_medium_a800.sh
#
#   # Typical detached launch
#   nohup bash scripts/run_v52_true_gt_v2_medium_a800.sh \
#       > outputs/ablations/v52_true_gt_v2_medium_a800_nohup.log 2>&1 &

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: project only uses GPUs 6/7 on A800. Default to 6, allow override.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_v2_standard.yaml \
    --num_domains 1 \
    --use_full_precision_dlt \
    --use_robust_dlt_reweight \
    --use_irls_reweight \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_dropout 0.2 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --v45_adaptive_weight_hidden 32 \
    --v45_adaptive_weight_n_layers 1 \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 \
    --v46_svg_min_views 2 \
    --v46_svg_hidden 64 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 \
    --v50_sefh_num_layers 2 \
    --v50_sefh_dropout 0.1 \
    --v50_sefh_loss_weight 0.0 \
    --v50_sefh_aleatoric_weight 0.0 \
    --v50_sefh_identity_init_gate \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 \
    --v51_cdsvr_num_heads 4 \
    --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_offset_min 0.05 \
    --v51_cdsvr_use_domain_label \
    --v51_cdsvr_uncertainty_temperature 1.0 \
    --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 \
    --v52_uwt_n_layers 2 \
    --v52_uwt_weight_type per_view_joint \
    --v52_uwt_temperature 1.0 \
    --v52_uwt_use_geometry_bias \
    --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init \
    --v52_uwt_min_weight 0.05 \
    --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 1e-4 \
    --num_workers 4 \
    --d 128 \
    --residual_hidden 256 \
    --n_st_layers 3 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 13 \
    --epochs 8 \
    --batch_size 16 \
    --train_samples 1024 \
    --val_stride 20 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
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
    --use_hierarchical_multiview_v30 \
    --v30_n_part_layers 2 \
    --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 \
    --v29_floor_loss_weight 0.01 \
    --v29_bone_temporal_weight 0.01 \
    --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 1 \
    --use_variable_view_training \
    --variable_view_min_views 2 \
    --variable_view_max_views 4 \
    --variable_view_max_views_start 4 \
    --variable_view_curriculum_alpha 2.0 \
    --variable_view_permute \
    --pa_loss_weight 0.5 \
    --monotonic_loss_weight 0.1 \
    --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 \
    --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 \
    --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 \
    --outlier_view_noise_std 15.0 \
    --output outputs/ablations/v52_true_gt_v2_medium_a800.pth \
    > outputs/ablations/v52_true_gt_v2_medium_a800.log 2>&1
