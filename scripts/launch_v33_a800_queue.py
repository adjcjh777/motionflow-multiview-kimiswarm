#!/usr/bin/env python3
"""Poll A800-D and launch the v32/v33 A800 queue when a GPU frees.

This script runs after the v31 top-5 variants.  It launches pending v32
ablations plus the new v33 modules (uncertainty-aware triangulation,
outlier-view rejection, ray-conditioned attention, and a combined run),
and later v46 sparse-view generalization, v47 temporal aggregation, and
v48 domain generalization runs.
"""

from __future__ import annotations

import re
import subprocess
import time


A800_REPO = "/mnt/nvme0n1p1/zhangzy/motionflow-multiview-kimiswarm-iter20"
SSH_HOST = "a800-D"
MIN_FREE_MIB = 30000  # d=64 full run on A800
POLL_INTERVAL = 60  # seconds

COMMON_FLAGS = (
    "--use_mixed_loader "
    "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
    "--use_full_precision_dlt --use_domain_embedding "
    "--use_deformable_cross_view_attention_v18 "
    "--use_multiview_geometry_fusion_v25 --v25_geom_loss_weight 0.1 --v25_dropout 0.2 "
    "--v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
    "--num_workers 4 --d 64 --residual_hidden 128 --n_st_layers 2 "
    "--graph_num_layers 1 --n_joint_layers 1 --n_heads 4 "
    "--clip_len 9 --epochs 20 --batch_size 8 --train_samples 1000 --val_stride 10 "
    "--lr 1e-3 --lr_cosine --lr_warmup_epochs 3 --lr_min 1e-6 --max_grad_norm 1.0 --ema_decay 0.999 "
    "--early_stopping_patience 5 --early_stopping_min_delta 0.001 "
    "--use_multiscale_fusion true --use_camera_conditioning true --use_epipolar_bias true "
    "--use_context_visibility true --use_skeleton_residual true --use_rotation_correction true "
    "--use_entropy_regularization true --attention_entropy_weight 0.01 "
    "--use_camera_view_embedding --use_set_view_aggregator "
    "--use_variable_view_training --variable_view_min_views 2 --variable_view_max_views 14 "
    "--variable_view_max_views_start 4 --variable_view_curriculum_alpha 2.0 --variable_view_permute "
    "--pa_loss_weight 0.5 --monotonic_loss_weight 0.1 --monotonic_margin 5.0 "
    "--reproj_loss_weight 0.1 --reproj_warmup_epochs 3 --aleatoric_reproj_loss_weight 0.1 "
    "--outlier_view_prob 0.3 --outlier_view_max_views 1 "
    "--outlier_view_offset_std 10.0 --outlier_view_noise_std 15.0 "
    "--use_hierarchical_multiview_v30 --v30_n_part_layers 2 --v30_stochastic_depth_prob 0.1 "
    "--use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.01 --v29_com_jitter_weight 0.001 "
    "--v29_physical_loss_warmup_epochs 3"
)

RUNS = [
    # v46/v47/v48 new-model stack (run first after the v25 baseline failed to converge).
    (
        "v46_sparse_view_generalization_on_v45",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v46_sparse_view_generalization_on_v45_a800",
    ),
    (
        "v47_temporal_aggregation_on_v46",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v47_temporal_aggregation --v47_temporal_d_model 64 --v47_temporal_n_heads 4 --v47_temporal_num_layers 2 --v47_temporal_loss_weight 0.01 --v47_use_view_count_conditioning "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v47_temporal_aggregation_on_v46_a800",
    ),
    (
        "v48_domain_generalization_on_v47",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v47_temporal_aggregation --v47_temporal_d_model 64 --v47_temporal_n_heads 4 --v47_temporal_num_layers 2 --v47_temporal_loss_weight 0.01 --v47_use_view_count_conditioning "
        "--use_v48_domain_generalization --v48_dg_hidden 64 --v48_dg_grl_lambda 0.1 --v48_dg_use_domain_film --v48_dg_use_ddwl --v48_dg_ddwl_temperature 2.0 --v48_dg_ddwl_warmup_epochs 1 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v48_domain_generalization_on_v47_a800",
    ),
    # v49-Lite: causal Conv1D temporal aggregation instead of the v47 transformer.
    (
        "v49_lite_temporal_on_v46",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v49_lite_temporal_aggregation --v49_lite_temporal_d_model 32 --v49_lite_temporal_num_layers 2 --v49_lite_temporal_kernel_size 3 --v49_lite_temporal_loss_weight 0.01 --v49_lite_temporal_use_view_count_conditioning "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v49_lite_temporal_on_v46_a800",
    ),
    # v49-Lite scaled: larger model + 10k samples to stress-test temporal aggregation.
    (
        "v49_lite_temporal_on_v46_scaled",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v49_lite_temporal_aggregation --v49_lite_temporal_d_model 32 --v49_lite_temporal_num_layers 2 --v49_lite_temporal_kernel_size 3 --v49_lite_temporal_loss_weight 0.01 --v49_lite_temporal_use_view_count_conditioning "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 10 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v49_lite_temporal_on_v46_scaled_a800",
    ),
    # v50 self-evolution feedback head on top of v46 sparse-view generalization.
    (
        "v50_self_evolution_feedback_head_on_v46",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v50_self_evolution_feedback_head --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_loss_weight 0.01 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 10 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v50_self_evolution_feedback_head_on_v46_a800",
    ),
    # v50 ablations: domain coupling, v49-Lite temporal coupling, and loss-weight sweep.
    (
        "v50_self_evolution_feedback_head_on_v48",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v47_temporal_aggregation --v47_temporal_d_model 64 --v47_temporal_n_heads 4 --v47_temporal_num_layers 2 --v47_temporal_loss_weight 0.01 --v47_use_view_count_conditioning "
        "--use_v48_domain_generalization --v48_dg_hidden 64 --v48_dg_grl_lambda 0.1 --v48_dg_use_domain_film --v48_dg_use_ddwl --v48_dg_ddwl_temperature 2.0 --v48_dg_ddwl_warmup_epochs 1 "
        "--use_v50_self_evolution_feedback_head --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_loss_weight 0.01 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 10 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v50_self_evolution_feedback_head_on_v48_a800",
    ),
    (
        "v50_self_evolution_feedback_head_on_v49_lite",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v49_lite_temporal_aggregation --v49_lite_temporal_d_model 32 --v49_lite_temporal_num_layers 2 --v49_lite_temporal_kernel_size 3 --v49_lite_temporal_loss_weight 0.01 --v49_lite_temporal_use_view_count_conditioning "
        "--use_v50_self_evolution_feedback_head --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_loss_weight 0.01 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 10 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v50_self_evolution_feedback_head_on_v49_lite_a800",
    ),
    (
        "v50_self_evolution_feedback_head_low_loss",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--use_v46_sparse_view_generalization --v46_svg_view_dropout_prob 0.3 --v46_svg_min_views 2 --v46_svg_hidden 64 --v46_svg_use_curriculum "
        "--use_v50_self_evolution_feedback_head --v50_sefh_hidden 64 --v50_sefh_num_layers 2 --v50_sefh_loss_weight 0.001 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 10 --early_stopping_patience 3 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v50_self_evolution_feedback_head_low_loss_a800",
    ),
    # Legacy v42/v43 ablations (queued after the new v45-v49 stack).
    (
        "v42_v36_physical_domain_no_v37",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 "
        "--domain_loss_weights 1.0,1.5",
        "omniview_fusion_v42_v36_physical_domain_no_v37_a800",
    ),
    (
        "v43_adaptive_node_residual_on_v42",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 "
        "--domain_loss_weights 1.0,1.5 --use_v43_adaptive_node_residual",
        "omniview_fusion_v43_adaptive_node_residual_on_v42_a800",
    ),
    (
        "v43_adaptive_node_residual_scaled",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 --v36_ugigr_dropout 0.1 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 "
        "--domain_loss_weights 1.0,1.5 --use_v43_adaptive_node_residual "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 10000 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v43_adaptive_node_residual_scaled_a800",
    ),
    (
        "v43_adaptive_node_residual_all_train",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 --v36_ugigr_dropout 0.1 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 "
        "--domain_loss_weights 1.0,1.5 --use_v43_adaptive_node_residual "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v43_adaptive_node_residual_all_train_a800",
    ),
    # v45 adaptive geometry fusion (learnable triangulation weights) on top of v25 h36m+mpi.
    (
        "v45_adaptive_geometry_fusion_h36m_mpi",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val.yaml "
        "--use_multiview_geometry_fusion_v25 --v25_use_geometry_attention --v25_use_learned_depth_triangulation --v25_use_geometry_bundle_adjustment "
        "--use_v45_adaptive_geometry_fusion --v45_adaptive_weight_type per_view_joint --v45_adaptive_weight_hidden 32 --v45_adaptive_weight_n_layers 1 "
        "--d 128 --residual_hidden 256 --n_st_layers 3 --batch_size 16 --clip_len 13 --train_samples 200 --epochs 5 --early_stopping_patience 2 --early_stopping_min_delta 0.001 --weight_decay 1e-4",
        "omniview_fusion_v45_adaptive_geometry_fusion_all_train_a800",
    ),
    # Pending v31 ablation.
    (
        "v31_physical_floor_only",
        "--use_hierarchical_multiview_v30 --use_physical_space_temporal_loss_v29 --v29_floor_loss_weight 0.01 --v29_bone_temporal_weight 0.0 --v29_com_jitter_weight 0.0 --v29_physical_loss_warmup_epochs 3",
        "omniview_fusion_v31_physical_floor_only_a800",
    ),
    # v32 ablations.
    (
        "v32_domain_aware_view_curriculum",
        "--domain_aware_view_curriculum",
        "omniview_fusion_v32_domain_aware_view_curriculum_a800",
    ),
    (
        "v32_trajectory_consistency_refiner",
        "--use_trajectory_consistency_v32 --v32_smooth_weight 1e-3 --v32_drift_weight 1e-2",
        "omniview_fusion_v32_trajectory_consistency_a800",
    ),
    (
        "v32_combined",
        "--domain_aware_view_curriculum --use_trajectory_consistency_v32 --v32_smooth_weight 1e-3 --v32_drift_weight 1e-2",
        "omniview_fusion_v32_combined_a800",
    ),
    (
        "v32_ray_attention",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --v31_use_ray_attention",
        "omniview_fusion_v32_ray_attention_a800",
    ),
    (
        "v32_physical_alignment",
        "--use_physical_space_alignment_v32 --v28_floor_loss_weight 0.01 --v28_bone_temporal_weight 0.01",
        "omniview_fusion_v32_physical_alignment_a800",
    ),
    # v33 new modules.
    (
        "v33_uncertainty_aware_triangulation",
        "--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01",
        "omniview_fusion_v33_uncertainty_aware_triangulation_a800",
    ),
    (
        "v33_outlier_view_rejection",
        "--use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.1 --outlier_view_prob 0.3 --outlier_view_max_views 1",
        "omniview_fusion_v33_outlier_view_rejection_a800",
    ),
    (
        "v33_ray_conditioned_attention",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0",
        "omniview_fusion_v33_ray_conditioned_attention_a800",
    ),
    (
        "v33_combined_all_three",
        "--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 "
        "--use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.1 "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0 "
        "--outlier_view_prob 0.3 --outlier_view_max_views 1",
        "omniview_fusion_v33_combined_all_three_a800",
    ),
    # v33 hierarchical multi-scale spatial pyramid.
    (
        "v33_hierarchical_multiscale_spatial_pyramid",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4",
        "omniview_fusion_v33_hierarchical_multiscale_spatial_pyramid_a800",
    ),
    # v33 combined fixed (lower outlier supervised weight + weight decay).
    (
        "v33_combined_all_three_fixed",
        "--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 "
        "--use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.01 --outlier_view_prob 0.3 --outlier_view_max_views 1 "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0 "
        "--weight_decay 1e-4",
        "omniview_fusion_v33_combined_all_three_fixed_a800",
    ),
    # v34 view-joint graph network.
    (
        "v34_view_joint_graph_network",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4",
        "omniview_fusion_v34_view_joint_graph_network_a800",
    ),
    # v34 geometry-aware view-joint graph network.
    (
        "v34_geometry_view_joint_graph_network",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4",
        "omniview_fusion_v34_geometry_view_joint_graph_network_a800",
    ),
    # v34 geometry-aware VJGN ablations.
    (
        "v34_geometry_view_joint_graph_network_n_layers_1",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 1 --v34_gvjgn_n_heads 4",
        "omniview_fusion_v34_geometry_view_joint_graph_network_n_layers_1_a800",
    ),
    (
        "v34_geometry_view_joint_graph_network_dropout_0_1",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4 --v34_gvjgn_dropout 0.1",
        "omniview_fusion_v34_geometry_view_joint_graph_network_dropout_0_1_a800",
    ),
    # v33 HMSP + v34 geometry-aware VJGN stack.
    (
        "v34_hmsp_geometry_vjgn_stack",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4 "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4",
        "omniview_fusion_v34_hmsp_geometry_vjgn_stack_a800",
    ),
    # v33 HMSP + v34 VJGN stack.
    (
        "v34_hmsp_vjgn_stack",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4 "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4",
        "omniview_fusion_v34_hmsp_vjgn_stack_a800",
    ),
    # v33 combined + HMSP (maximal v33 stack).
    (
        "v33_combined_all_three_plus_hmsp",
        "--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 "
        "--use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.1 "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0 "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4 "
        "--outlier_view_prob 0.3 --outlier_view_max_views 1",
        "omniview_fusion_v33_combined_all_three_plus_hmsp_a800",
    ),
    # v34 geometry-aware VJGN on top of v33 combined fixed (maximal v34 stack).
    (
        "v34_geometry_vjgn_combined_fixed_max",
        "--use_uncertainty_aware_triangulation_v33 --v33_uat_loss_weight 0.01 "
        "--use_outlier_view_rejection_v33 --v33_outlier_z_thresh 3.0 --v33_outlier_soft_beta 1.0 --v33_outlier_supervised_weight 0.01 "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias --use_ray_conditioned_attention_v33 --v33_n_heads 4 --v33_n_layers 2 --v33_use_ray_bias --v33_residual_gate_init -6.0 "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4 "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4 "
        "--outlier_view_prob 0.3 --outlier_view_max_views 1 --weight_decay 1e-4",
        "omniview_fusion_v34_geometry_vjgn_combined_fixed_max_a800",
    ),
    # v35 temporal view-joint graph network on top of v34 VJGN.
    (
        "v35_temporal_vjgn_on_v34_vjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4",
        "omniview_fusion_v35_temporal_vjgn_on_v34_vjgn_a800",
    ),
    # v35 temporal view-joint graph network on top of v34 geometry-aware VJGN.
    (
        "v35_temporal_vjgn_on_v34_geometry_vjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4",
        "omniview_fusion_v35_temporal_vjgn_on_v34_geometry_vjgn_a800",
    ),
    # v36 uncertainty-gated iterative graph refinement on top of v34 VJGN.
    (
        "v36_ugigr_on_v34_vjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64",
        "omniview_fusion_v36_ugigr_on_v34_vjgn_a800",
    ),
    # v36 uncertainty-gated iterative graph refinement on top of v35 TVJGN.
    (
        "v36_ugigr_on_v35_tvjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64",
        "omniview_fusion_v36_ugigr_on_v35_tvjgn_a800",
    ),
    # v36 on top of the strongest v34 stack (HMSP + geometry-aware VJGN).
    (
        "v36_ugigr_on_v34_hmsp_geometry_vjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_hierarchical_multiscale_spatial_pyramid_v33 --v33_hmsp_scales 1 2 4 "
        "--use_geometry_view_joint_graph_network_v34 --v34_gvjgn_n_layers 2 --v34_gvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64",
        "omniview_fusion_v36_ugigr_on_v34_hmsp_geometry_vjgn_a800",
    ),
    # v36 ablation: only 1 iterative step on top of v34 VJGN.
    (
        "v36_ugigr_n_iters_1_on_v34_vjgn",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 1 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64",
        "omniview_fusion_v36_ugigr_n_iters_1_on_v34_vjgn_a800",
    ),
    # v37 self-critique view reliability on top of v36 UGIGR.
    (
        "v37_scvr_on_v36_ugigr",
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_self_critique_view_reliability_v37 --v37_scvr_hidden 64 --v37_scvr_n_layers 2 --v37_scvr_use_temporal_context --v37_scvr_loss_weight 0.01",
        "omniview_fusion_v37_scvr_on_v36_ugigr_a800",
    ),
    # v38: v37 + expanded WebBridge training data.
    (
        "v38_expanded_data_scvr",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_self_critique_view_reliability_v37 --v37_scvr_hidden 64 --v37_scvr_n_layers 2 --v37_scvr_use_temporal_context --v37_scvr_loss_weight 0.01",
        "omniview_fusion_v38_expanded_data_scvr_a800",
    ),
    # v39: v38 + reliability-coupled adaptive graph refinement (RCAR).
    (
        "v39_rcgr_on_v38_scvr",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_self_critique_view_reliability_v37 --v37_scvr_hidden 64 --v37_scvr_n_layers 2 --v37_scvr_use_temporal_context --v37_scvr_loss_weight 0.01 "
        "--use_reliability_coupled_graph_refinement_v39",
        "omniview_fusion_v39_rcgr_on_v38_scvr_a800",
    ),
    # v40: v39 + skeleton-aware physical loss.
    (
        "v40_skeleton_physical_loss_on_v39_rcgr",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_self_critique_view_reliability_v37 --v37_scvr_hidden 64 --v37_scvr_n_layers 2 --v37_scvr_use_temporal_context --v37_scvr_loss_weight 0.01 "
        "--use_reliability_coupled_graph_refinement_v39 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02",
        "omniview_fusion_v40_skeleton_physical_loss_on_v39_rcgr_a800",
    ),
    # v41: per-domain weighted MSE for mixed WebBridge/H36M/MPI training.
    (
        "v41_domain_weighted_loss_on_v40",
        "--mixed_manifest configs/splits/webbridge_h36m_mpi_mixed_train_val_expanded.yaml "
        "--use_hierarchical_multiview_v31 --v31_geometry_bias "
        "--use_view_joint_graph_network_v34 --v34_vjgn_n_layers 2 --v34_vjgn_n_heads 4 "
        "--use_temporal_view_joint_graph_network_v35 --v35_tvjgn_n_layers 2 --v35_tvjgn_n_heads 4 "
        "--use_uncertainty_gated_iterative_graph_refinement_v36 --v36_ugigr_n_layers 1 --v36_ugigr_n_iters 2 --v36_ugigr_n_heads 4 --v36_ugigr_uncertainty_hidden 64 "
        "--use_self_critique_view_reliability_v37 --v37_scvr_hidden 64 --v37_scvr_n_layers 2 --v37_scvr_use_temporal_context --v37_scvr_loss_weight 0.01 "
        "--use_reliability_coupled_graph_refinement_v39 "
        "--use_skeleton_physical_loss_v40 --v40_bone_weight 0.05 --v40_joint_limit_weight 0.01 --v40_symmetry_weight 0.02 --v40_floor_weight 0.02 "
        "--domain_loss_weights 1.0,1.5",
        "omniview_fusion_v41_domain_weighted_loss_on_v40_a800",
    ),
]


def a800_ssh(cmd: str) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", SSH_HOST, cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )


def sync_repo_to_a800() -> None:
    """Push the local working tree to A800-D.

    GitHub access from A800 is intermittent, so we sync by archiving the local
    tracked files and extracting them on A800.  The local directory is the source
    of truth for code; outputs, .git, virtualenvs, and caches are left untouched
    on the remote.
    """
    import pathlib
    import tempfile

    local_root = pathlib.Path(__file__).resolve().parent.parent
    archive = pathlib.Path(tempfile.gettempdir()) / "motionflow_a800_sync.tar.gz"
    print("Syncing local repo to A800-D via git archive + tar...")
    # Archive tracked files at HEAD.
    subprocess.check_output(
        ["git", "-C", str(local_root), "archive", "--format=tar.gz", "-o", str(archive), "HEAD"],
        stderr=subprocess.STDOUT,
    )
    # Copy archive to A800 and extract in place.
    subprocess.check_output(
        ["scp", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", str(archive), f"{SSH_HOST}:{A800_REPO}/.sync.tar.gz"],
        stderr=subprocess.STDOUT,
    )
    a800_ssh(f"cd {A800_REPO} && tar -xzf .sync.tar.gz && rm .sync.tar.gz && find . -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null || true")
    print("Archive sync complete.")


def gpu_free_mibs() -> list[tuple[int, int]]:
    out = a800_ssh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    pairs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def used_gpus_from_tmux() -> set[int]:
    """Return GPU indices already used by v31/v32/v33/v34/v35/v36/v47/v48 tmux sessions.

    Note: this is kept for diagnostics only.  The poller now relies on free
    memory rather than tmux names, so it can co-locate queued runs alongside
    the long-running v31 top-5 sessions when sufficient memory is available.
    """
    gpus: set[int] = set()
    try:
        out = a800_ssh("tmux ls 2>/dev/null || true")
    except subprocess.CalledProcessError:
        return gpus
    for line in out.splitlines():
        # v31_top5_<name>_gpuN, v32_<name>_gpuN, ..., v39_<name>_gpuN, v46_<name>_gpuN, v47_<name>_gpuN, v48_<name>_gpuN
        match = re.search(r"((?:v25_|v31_top5_|v31_|v32_|v33_|v34_|v35_|v36_|v37_|v38_|v39_|v42_|v43_|v44_|v45_|v46_|v47_|v48_)[a-zA-Z0-9_]+)_gpu(\d+):", line)
        if match:
            gpus.add(int(match.group(2)))
    return gpus


def running_run_names() -> set[str]:
    """Return the set of run keys already running on A800."""
    names: set[str] = set()
    try:
        out = a800_ssh("tmux ls 2>/dev/null || true")
    except subprocess.CalledProcessError:
        return names
    for line in out.splitlines():
        match = re.search(r"((?:v25_|v31_top5_|v31_|v32_|v33_|v34_|v35_|v36_|v37_|v38_|v39_|v42_|v43_|v44_|v45_|v46_|v47_|v48_)[a-zA-Z0-9_]+)_gpu\d+:", line)
        if match:
            names.add(match.group(1))
    return names


def launch_run(name: str, extra_flags: str, output: str, gpu: int) -> None:
    session = f"{name}_gpu{gpu}"
    cmd = (
        f"cd {A800_REPO} && "
        f"CUDA_VISIBLE_DEVICES={gpu} python3 -u experiments/train_omniview_fusion_v5_webbridge_multi.py "
        f"{COMMON_FLAGS} {extra_flags} "
        f"--output outputs/{output}.pth "
        f"> outputs/{output}.log 2>&1"
    )
    a800_ssh(f"tmux has-session -t {session} 2>/dev/null || tmux new-session -d -s {session} -n v33 '{cmd}'")
    print(f"Launched {session} (GPU {gpu}) for {name}")


def main() -> None:
    queue = list(RUNS)
    try:
        sync_repo_to_a800()
    except subprocess.CalledProcessError as e:
        print(f"Warning: rsync to A800 failed: {e}; continuing with existing A800 repo state")
    tmux_gpus = used_gpus_from_tmux()
    already_running = running_run_names()
    queue = [(n, f, o) for n, f, o in queue if n not in already_running]
    print(f"GPUs with tmux sessions (diagnostic): {tmux_gpus}")
    print(f"Already-running runs names: {already_running}")
    print(f"Remaining queue: {[n for n, _, _ in queue]}")

    # Track which GPUs have been assigned a run in this poller session to avoid
    # launching multiple runs on the same GPU in rapid succession.
    launched_gpus: set[int] = set()

    while queue:
        pairs = gpu_free_mibs()
        candidates = [
            (g, f) for g, f in pairs
            if f >= MIN_FREE_MIB and g not in launched_gpus
        ]
        if not candidates:
            # All candidate GPUs are busy; reset so we can try again next round
            # after a full polling interval has passed.
            launched_gpus.clear()
            print(f"No GPU with >= {MIN_FREE_MIB} MiB free; sleeping {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue
        # Prefer the GPU with the most free memory.
        gpu, free_mib = max(candidates, key=lambda x: x[1])
        launched_gpus.add(gpu)
        name, extra_flags, output = queue.pop(0)
        launch_run(name, extra_flags, output, gpu)
        print(f"GPU {gpu} has {free_mib} MiB free; launched {name}")
        time.sleep(60)
    print("All v32/v33/v34/v35/v36/v45/v46/v47/v48 runs launched.")


if __name__ == "__main__":
    main()
