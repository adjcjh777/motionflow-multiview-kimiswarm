#!/usr/bin/env bash
# Local RTX 4090 medium run for v51 Cross-Domain Sparse-View Reliability
# (CDSVR) on top of v50 Self-Evolution Feedback Head.
#
# 200 samples / 5 epochs to validate that the stable MSE-based v50/v51
# auxiliary losses scale beyond the tiny smoke. See issue #181.
set -euo pipefail

python -u experiments/train_omniview_fusion_v5_webbridge_multi.py \
    --use_mixed_loader \
    --mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml \
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
    --num_workers 0 \
    --d 64 \
    --residual_hidden 128 \
    --n_st_layers 2 \
    --graph_num_layers 1 \
    --n_joint_layers 1 \
    --n_heads 4 \
    --clip_len 9 \
    --epochs 5 \
    --batch_size 4 \
    --train_samples 200 \
    --val_stride 10 \
    --lr 1e-3 \
    --lr_cosine \
    --lr_warmup_epochs 1 \
    --lr_min 1e-6 \
    --max_grad_norm 1.0 \
    --ema_decay 0.999 \
    --early_stopping_patience 2 \
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
    --variable_view_max_views 8 \
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
    --use_hierarchical_multiview_v30 \
    --v30_n_part_layers 2 \
    --v30_stochastic_depth_prob 0.1 \
    --use_physical_space_temporal_loss_v29 \
    --v29_floor_loss_weight 0.01 \
    --v29_bone_temporal_weight 0.01 \
    --v29_com_jitter_weight 0.001 \
    --v29_physical_loss_warmup_epochs 1 \
    --output outputs/omniview_fusion_v51_cdsvr_medium_local_4090.pth \
    >> outputs/omniview_fusion_v51_cdsvr_medium_local_4090.log 2>&1
