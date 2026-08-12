#!/usr/bin/env bash
# A800-D medium run for v57 Domain-Conditional Physical-Space Calibration
# (DC-PSC) on the NON-CIRCULAR H36M true-GT standard protocol
# (S1,5,6,7,8 train -> S9/S11 val).
#
# This script replicates the exact hyperparameters of the local RTX 4090 run
# scripts/run_v57_h36m_true_gt_medium.sh, which produced a true best val MPJPE
# of 75.16 mm at epoch 3. The only intended change is the fixed trainer
# (motionflow_mv/training/trainer_v2.py), which now monitors ``mpjpe`` instead
# of ``loss`` for best-checkpoint selection.
#
# Usage
# -----
#   # Default: GPU 4 (override with CUDA_VISIBLE_DEVICES if busy)
#   bash scripts/run_v57_true_gt_medium_a800.sh
#
#   # Run on a specific free GPU
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_v57_true_gt_medium_a800.sh

set -euo pipefail

# Pin to the A800-D repo root so the script can be launched from anywhere.
cd /mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20

# GPU discipline: default to GPU 4, but allow CUDA_VISIBLE_DEVICES override.
# GPU 4/6 are currently occupied by v25 ablations; check nvidia-smi first.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES

# Use the project venv Python by default; allow override.
PYTHON=${PYTHON:-/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm/.venv/bin/python}

mkdir -p outputs/ablations

$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
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
    --use_v57_domain_conditional_psc \
    --v57_dcpsc_hidden 64 \
    --v57_dcpsc_n_layers 2 \
    --v57_dcpsc_num_domains 8 \
    --v57_dcpsc_use_floor \
    --v57_dcpsc_use_bone_scale \
    --v57_dcpsc_use_uwt_weights \
    --v57_dcpsc_identity_init \
    --v57_dcpsc_residual_gate_init -6.0 \
    --v57_dcpsc_loss_weight 0.1 \
    --v57_dcpsc_floor_weight 0.01 \
    --v57_dcpsc_bone_weight 0.1 \
    --v57_dcpsc_reproj_weight 0.1 \
    --v57_dcpsc_warmup_epochs 1 \
    --v57_dcpsc_min_visible_views 2 \
    --num_workers 0 \
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
    --output outputs/ablations/v57_true_gt_medium_a800.pth \
    > outputs/ablations/v57_true_gt_medium_a800.log 2>&1
