#!/usr/bin/env bash
# Local RTX 4090 smoke for v80 on the NON-CIRCULAR H36M true-GT standard
# protocol (S1,5,6,7,8 train -> S9/S11 val; issue #194). Verifies the
# end-to-end wiring of the regenerated metre-convention npz
# (data/h36m_true_gt/*_m.npz) with the mixed loader before committing to
# long runs.
#
# Sanity anchors: DLT on these labels = S9 29.54 / S11 21.81 mm MPJPE
# (data/h36m_true_gt/dlt_baseline_h36m.json). A wired-up model should land
# val MPJPE on the same order of magnitude (tens of mm), NOT ~430 mm
# (metre/mm confusion) and NOT sub-mm (label leakage).
set -euo pipefail

PYTHON=${PYTHON:-python}
$PYTHON -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/h36m_true_gt_standard.yaml \
    --use_domain_embedding \
    --use_deformable_cross_view_attention_v18 \
    --use_multiview_geometry_fusion_v25 \
    --v25_geom_loss_weight 0.1 \
    --v25_use_geometry_attention \
    --v25_use_learned_depth_triangulation \
    --v25_use_geometry_bundle_adjustment \
    --use_v45_adaptive_geometry_fusion \
    --v45_adaptive_weight_type per_view_joint \
    --use_v46_sparse_view_generalization \
    --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 \
    --v46_svg_use_curriculum \
    --use_v50_self_evolution_feedback_head \
    --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_dropout 0.1 \
    --v50_sefh_identity_init_gate \
    --v50_sefh_loss_weight 0.0 --v50_sefh_aleatoric_weight 0.0 \
    --use_v51_cross_domain_sparse_view_reliability \
    --v51_cdsvr_hidden 64 --v51_cdsvr_num_heads 4 --v51_cdsvr_dropout 0.1 \
    --v51_cdsvr_use_domain_label --v51_cdsvr_identity_init_gate \
    --v51_cdsvr_loss_weight 0.0 \
    --use_v52_uncertainty_weighted_triangulation \
    --v52_uwt_hidden 64 --v52_uwt_n_layers 2 --v52_uwt_weight_type per_view_joint \
    --v52_uwt_use_geometry_bias --v52_uwt_use_feature_bias \
    --v52_uwt_identity_init --v52_uwt_min_weight 0.05 --v52_uwt_loss_weight 0.01 \
    --v52_uwt_damping 0.0001 \
    --use_v80_view_reliability \
    --v80_vrbt_hidden 64 --v80_vrbt_n_layers 2 \
    --v80_vrbt_weight_type per_view_joint \
    --v80_vrbt_use_geometry_bias --v80_vrbt_use_feature_bias \
    --v80_vrbt_identity_init --v80_vrbt_min_weight 0.05 \
    --num_workers 0 \
    --d 64 --residual_hidden 128 --n_st_layers 2 \
    --graph_num_layers 1 --n_joint_layers 1 --n_heads 4 \
    --epochs 2 --batch_size 8 --train_samples 128 --val_stride 200 \
    --lr 1e-3 --lr_cosine --lr_warmup_epochs 1 --lr_min 1e-6 \
    --max_grad_norm 1.0 --ema_decay 0.999 \
    --use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true \
    --use_context_visibility true --use_skeleton_residual true --use_rotation_correction true \
    --use_entropy_regularization true --attention_entropy_weight 0.01 \
    --use_camera_view_embedding --use_set_view_aggregator \
    --use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 4 \
    --variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute \
    --pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 \
    --reproj_loss_weight 0.1 --reproj_warmup_epochs 1 \
    --aleatoric_reproj_loss_weight 0.1 \
    --outlier_view_prob 0.3 --outlier_view_max_views 1 \
    --outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 \
    --output outputs/omniview_fusion_v80_h36m_true_gt_smoke.pth \
    > outputs/omniview_fusion_v80_h36m_true_gt_smoke.log 2>&1
