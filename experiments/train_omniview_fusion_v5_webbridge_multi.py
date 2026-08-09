"""Train OmniMultiViewFusionV5 on one or more WebBridge datasets.

This script mirrors ``train_omniview_fusion_v4_webbridge_multi.py`` but targets
``OmniMultiViewFusionV5``.  It adds v5-specific capabilities:

* camera-conditioned view embeddings (permutation invariant over view order),
* set-transformer view aggregation,
* variable-view training with random view subsets and permutations.

Usage
-----
    # CPU smoke (1 epoch, synthetic data, no external files)
    python experiments/train_omniview_fusion_v5_webbridge_multi.py --smoke

    # Full training on a WebBridge manifest
    python experiments/train_omniview_fusion_v5_webbridge_multi.py \
        --manifest configs/splits/webbridge_h36m_train_val.yaml \
        --use_camera_view_embedding --use_set_view_aggregator \
        --d 128 --residual_hidden 128 --epochs 30
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_cameras  # noqa: E402
from motionflow_mv.data.split_loader import (  # noqa: E402
    load_multi_dataset_manifest,
    load_split_manifest,
)
from motionflow_mv.data.webbridge_mixed_dataset import (  # noqa: E402
    build_webbridge_mixed_dataloaders,
    webbridge_mixed_collate_fn,
)
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (  # noqa: E402
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)
from motionflow_mv.fusion.omniview_fusion_v5 import (  # noqa: E402
    OmniMultiViewFusionV5,
)
from motionflow_mv.losses.kinematic_v15 import (  # noqa: E402
    joint_limit_loss,
    temporal_bone_length_loss,
)
from motionflow_mv.losses.skeleton_physical_loss_v40 import (  # noqa: E402
    SkeletonPhysicalLossV40,
)
from motionflow_mv.losses.procrustes_loss import procrustes_mse_loss  # noqa: E402
from motionflow_mv.fusion.self_evolution_feedback_head_v50 import (  # noqa: E402
    compute_sefh_loss,
)
from motionflow_mv.training.trainer_v2 import TrainerV2, build_lr_scheduler  # noqa: E402


def build_model_from_args(
    args: Namespace,
    n_joints: int,
    n_views: int,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Build and return an ``OmniMultiViewFusionV5`` model from CLI args."""

    model_kwargs: Dict[str, Any] = {
        "j": n_joints,
        "d": args.d,
        "n_views": n_views,
        "n_heads": args.n_heads,
        "n_joint_layers": args.n_joint_layers,
        "n_st_layers": args.n_st_layers,
        "residual_hidden": args.residual_hidden,
        "graph_num_layers": args.graph_num_layers,
        "epipolar_loss_weight": args.epipolar_loss_weight,
        "return_pp_delta": False,
        "return_covariance": True,
        # v4 toggles
        "use_multiscale_fusion": args.use_multiscale_fusion,
        "use_adaptive_multiscale_fusion": getattr(args, "use_adaptive_multiscale_fusion", False),
        "use_camera_conditioning": args.use_camera_conditioning,
        "use_epipolar_bias": args.use_epipolar_bias,
        "use_context_visibility": args.use_context_visibility,
        "use_skeleton_residual": args.use_skeleton_residual,
        "use_skeleton_residual_v31": args.use_skeleton_residual_v31,
        "use_kinematic_refiner": args.use_kinematic_refiner,
        "use_adaptive_view_selection": args.use_adaptive_view_selection,
        "use_rotation_correction": args.use_rotation_correction,
        "use_entropy_regularization": args.use_entropy_regularization,
        # v5 toggles
        "use_camera_view_embedding": args.use_camera_view_embedding,
        "use_camera_view_embedding_v31": args.use_camera_view_embedding_v31,
        "use_set_view_aggregator": args.use_set_view_aggregator,
        "camera_view_embedding_hidden": args.camera_view_embedding_hidden,
        "set_view_n_isab_layers": args.set_view_n_isab_layers,
        "set_view_num_inducing_points": args.set_view_num_inducing_points,
        "set_view_dropout": args.set_view_dropout,
        # v6 toggles
        "use_perceiver_aggregator": args.use_perceiver_aggregator,
        "perceiver_n_latents": args.perceiver_n_latents,
        "perceiver_n_layers": args.perceiver_n_layers,
        "perceiver_n_heads": args.perceiver_n_heads,
        "perceiver_dropout": args.perceiver_dropout,
        # v17/v18 toggles
        "use_cross_view_transformer_v17": getattr(args, "use_cross_view_transformer_v17", False),
        "use_deformable_cross_view_attention_v18": getattr(args, "use_deformable_cross_view_attention_v18", False),
        "deformable_attention_use_topk_st": getattr(args, "deformable_attention_use_topk_st", False),
        # v19 toggles
        "use_temporal_perceiver_v19": getattr(args, "use_temporal_perceiver_v19", False),
        # v20 toggles
        "use_diffusion_refiner_v20": getattr(args, "use_diffusion_refiner_v20", False),
        "num_diffusion_steps": getattr(args, "num_diffusion_steps", 10),
        # v21 toggles
        "use_neural_bundle_adjustment_v21": getattr(args, "use_neural_bundle_adjustment_v21", False),
        # v22 toggles
        "use_kinematic_anthropometric_prior_v22": getattr(args, "use_kinematic_anthropometric_prior_v22", False),
        "kap_loss_weight": getattr(args, "kap_loss_weight", 0.01),
        "kap_use_angle_limit": getattr(args, "kap_use_angle_limit", True),
        "kap_max_flexion_deg": getattr(args, "kap_max_flexion_deg", 160.0),
        # v25 toggles
        "use_multiview_geometry_fusion_v25": getattr(args, "use_multiview_geometry_fusion_v25", False),
        "v25_geom_loss_weight": getattr(args, "v25_geom_loss_weight", 0.1),
        "v25_use_geometry_attention": getattr(args, "v25_use_geometry_attention", True),
        "v25_use_learned_depth_triangulation": getattr(args, "v25_use_learned_depth_triangulation", True),
        "v25_use_geometry_bundle_adjustment": getattr(args, "v25_use_geometry_bundle_adjustment", True),
        "v25_use_camera_joint_graph": getattr(args, "v25_use_camera_joint_graph", False),
        "v25_use_outlier_view_detector": getattr(args, "v25_use_outlier_view_detector", False),
        "v25_outlier_z_thresh": getattr(args, "v25_outlier_z_thresh", 3.0),
        "v25_outlier_soft_beta": getattr(args, "v25_outlier_soft_beta", 1.0),
        "v25_dropout": getattr(args, "v25_dropout", 0.1),
        "use_temporal_geometry_fusion_v26": getattr(args, "use_temporal_geometry_fusion_v26", False),
        "v26_temporal_window": getattr(args, "v26_temporal_window", 3),
        "v26_temporal_attention_residual_gate_init": getattr(args, "v26_temporal_attention_residual_gate_init", 0.0),
        "use_uncertainty_depth_proposals_v27": getattr(args, "use_uncertainty_depth_proposals_v27", False),
        "v27_uncertainty_loss_weight": getattr(args, "v27_uncertainty_loss_weight", 0.01),
        "v27_udp_n_mixtures": getattr(args, "v27_udp_n_mixtures", 1),
        # v45 adaptive geometry fusion for v25 triangulation
        "use_v45_adaptive_geometry_fusion": getattr(args, "use_v45_adaptive_geometry_fusion", False),
        "v45_adaptive_weight_type": getattr(args, "v45_adaptive_weight_type", "per_view"),
        "v45_adaptive_weight_hidden": getattr(args, "v45_adaptive_weight_hidden", 32),
        "v45_adaptive_weight_n_layers": getattr(args, "v45_adaptive_weight_n_layers", 1),
        # v46 sparse-view generalization
        "use_v46_sparse_view_generalization": getattr(args, "use_v46_sparse_view_generalization", False),
        "v46_svg_view_dropout_prob": getattr(args, "v46_svg_view_dropout_prob", 0.3),
        "v46_svg_min_views": getattr(args, "v46_svg_min_views", 2),
        "v46_svg_hidden": getattr(args, "v46_svg_hidden", 64),
        "v46_svg_use_curriculum": getattr(args, "v46_svg_use_curriculum", True),
        "use_physical_space_alignment_v28": getattr(args, "use_physical_space_alignment_v28", False),
        "use_physical_space_alignment_v32": getattr(args, "use_physical_space_alignment_v32", False),
        "v28_floor_loss_weight": getattr(args, "v28_floor_loss_weight", 0.0),
        "v28_bone_temporal_weight": getattr(args, "v28_bone_temporal_weight", 0.0),
        "use_test_time_self_evolution_v27": getattr(args, "use_test_time_self_evolution_v27", False),
        "v27_tte_n_iters": getattr(args, "v27_tte_n_iters", 3),
        "v27_tte_sigma_reproj": getattr(args, "v27_tte_sigma_reproj", 5.0),
        "v27_tte_residual_thresh_mm": getattr(args, "v27_tte_residual_thresh_mm", 0.5),
        # v29 toggles
        "use_hierarchical_multiview_v29": getattr(args, "use_hierarchical_multiview_v29", False),
        "v29_n_heads": getattr(args, "v29_n_heads", 4),
        "v29_n_part_layers": getattr(args, "v29_n_part_layers", 1),
        # v30 toggles
        "use_hierarchical_multiview_v30": getattr(args, "use_hierarchical_multiview_v30", False),
        "v30_n_heads": getattr(args, "v30_n_heads", 4),
        "v30_n_part_layers": getattr(args, "v30_n_part_layers", 1),
        "v30_dropout": getattr(args, "v30_dropout", 0.1),
        "v30_stochastic_depth_prob": getattr(args, "v30_stochastic_depth_prob", 0.0),
        "use_hierarchical_multiview_v31": getattr(args, "use_hierarchical_multiview_v31", False),
        "v31_geometry_bias": getattr(args, "v31_geometry_bias", True),
        "v31_use_ray_attention": getattr(args, "v31_use_ray_attention", False),
        # v33 hierarchical multi-scale spatial pyramid
        "use_hierarchical_multiscale_spatial_pyramid_v33": getattr(args, "use_hierarchical_multiscale_spatial_pyramid_v33", False),
        "v33_hmsp_scales": tuple(getattr(args, "v33_hmsp_scales", [1, 2, 4])),
        "v33_hmsp_n_heads": getattr(args, "v33_hmsp_n_heads", 4),
        "v33_hmsp_n_part_layers": getattr(args, "v33_hmsp_n_part_layers", 1),
        "v33_hmsp_dropout": getattr(args, "v33_hmsp_dropout", 0.1),
        "v33_hmsp_stochastic_depth_prob": getattr(args, "v33_hmsp_stochastic_depth_prob", 0.0),
        "v33_hmsp_use_geometry_bias": getattr(args, "v33_hmsp_use_geometry_bias", True),
        "v33_hmsp_use_adaptive_scale_fusion": getattr(args, "v33_hmsp_use_adaptive_scale_fusion", True),
        # v34 view-joint graph network
        "use_view_joint_graph_network_v34": getattr(args, "use_view_joint_graph_network_v34", False),
        "v34_vjgn_n_layers": getattr(args, "v34_vjgn_n_layers", 2),
        "v34_vjgn_n_heads": getattr(args, "v34_vjgn_n_heads", 4),
        "v34_vjgn_dropout": getattr(args, "v34_vjgn_dropout", 0.0),
        # v34 geometry-aware view-joint graph network
        "use_geometry_view_joint_graph_network_v34": getattr(args, "use_geometry_view_joint_graph_network_v34", False),
        "v34_gvjgn_n_layers": getattr(args, "v34_gvjgn_n_layers", 2),
        "v34_gvjgn_n_heads": getattr(args, "v34_gvjgn_n_heads", 4),
        "v34_gvjgn_dropout": getattr(args, "v34_gvjgn_dropout", 0.0),
        # v35 temporal view-joint graph network
        "use_temporal_view_joint_graph_network_v35": getattr(args, "use_temporal_view_joint_graph_network_v35", False),
        "v35_tvjgn_n_layers": getattr(args, "v35_tvjgn_n_layers", 2),
        "v35_tvjgn_n_heads": getattr(args, "v35_tvjgn_n_heads", 4),
        "v35_tvjgn_dropout": getattr(args, "v35_tvjgn_dropout", 0.0),
        # v36 uncertainty-gated iterative graph refinement
        "use_uncertainty_gated_iterative_graph_refinement_v36": getattr(args, "use_uncertainty_gated_iterative_graph_refinement_v36", False),
        "v36_ugigr_n_layers": getattr(args, "v36_ugigr_n_layers", 1),
        "v36_ugigr_n_iters": getattr(args, "v36_ugigr_n_iters", 2),
        "v36_ugigr_n_heads": getattr(args, "v36_ugigr_n_heads", 4),
        "v36_ugigr_dropout": getattr(args, "v36_ugigr_dropout", 0.0),
        "v36_ugigr_uncertainty_hidden": getattr(args, "v36_ugigr_uncertainty_hidden", 64),
        # v43 adaptive per-node residual on top of v36 UGIGR
        "use_v43_adaptive_node_residual": getattr(args, "use_v43_adaptive_node_residual", False),
        # v44 edge-type-aware uncertainty gating on top of v36 UGIGR
        "use_v44_edge_type_gate": getattr(args, "use_v44_edge_type_gate", False),
        # v37 self-critique view reliability estimator
        "use_self_critique_view_reliability_v37": getattr(args, "use_self_critique_view_reliability_v37", False),
        "v37_scvr_hidden": getattr(args, "v37_scvr_hidden", 64),
        "v37_scvr_n_layers": getattr(args, "v37_scvr_n_layers", 2),
        "v37_scvr_use_temporal_context": getattr(args, "v37_scvr_use_temporal_context", True),
        "v37_scvr_loss_weight": getattr(args, "v37_scvr_loss_weight", 0.01),
        # v39 reliability-coupled adaptive graph refinement
        "use_reliability_coupled_graph_refinement_v39": getattr(args, "use_reliability_coupled_graph_refinement_v39", False),
        # v32 temporal trajectory consistency
        "use_trajectory_consistency_v32": getattr(args, "use_traertainty_consistency_v32", False),
        "v32_smooth_weight": getattr(args, "v32_smooth_weight", 1e-3),
        "v32_drift_weight": getattr(args, "v32_drift_weight", 1e-2),
        # v33 uncertainty-aware triangulation
        "use_uncertainty_aware_triangulation_v33": getattr(args, "use_uncertainty_aware_triangulation_v33", False),
        "v33_uat_loss_weight": getattr(args, "v33_uat_loss_weight", 0.01),
        "v33_uat_log_var_min": getattr(args, "v33_uat_log_var_min", -10.0),
        "v33_uat_log_var_max": getattr(args, "v33_uat_log_var_max", 10.0),
        "v33_uat_covariance_hidden": getattr(args, "v33_uat_covariance_hidden", 64),
        # v33 outlier-view rejection
        "use_outlier_view_rejection_v33": getattr(args, "use_outlier_view_rejection_v33", False),
        "v33_outlier_z_thresh": getattr(args, "v33_outlier_z_thresh", 3.0),
        "v33_outlier_soft_beta": getattr(args, "v33_outlier_soft_beta", 1.0),
        "v33_outlier_use_feature_gate": getattr(args, "v33_outlier_use_feature_gate", True),
        "v33_outlier_use_part_scale": getattr(args, "v33_outlier_use_part_scale", True),
        "v33_outlier_use_domain_scale": getattr(args, "v33_outlier_use_domain_scale", True),
        "v33_outlier_supervised_weight": getattr(args, "v33_outlier_supervised_weight", 0.1),
        "v33_outlier_feature_hidden": getattr(args, "v33_outlier_feature_hidden", 64),
        # v33 ray-conditioned cross-view attention
        "use_ray_conditioned_attention_v33": getattr(args, "use_ray_conditioned_attention_v33", False),
        "v33_n_heads": getattr(args, "v33_n_heads", 4),
        "v33_n_layers": getattr(args, "v33_n_layers", 2),
        "v33_dropout": getattr(args, "v33_dropout", 0.1),
        "v33_use_ray_bias": getattr(args, "v33_use_ray_bias", True),
        "v33_residual_gate_init": getattr(args, "v33_residual_gate_init", -6.0),
        "use_test_time_self_evolution_v29": getattr(args, "use_test_time_self_evolution_v29", False),
        "v29_tte_n_iters": getattr(args, "v29_tte_n_iters", 3),
        "v29_tte_sigma_reproj": getattr(args, "v29_tte_sigma_reproj", 5.0),
        "v29_tte_residual_thresh_mm": getattr(args, "v29_tte_residual_thresh_mm", 0.5),
        "v29_tte_use_physical_space_alignment": getattr(args, "v29_tte_use_physical_space_alignment", True),
        "use_physical_space_temporal_loss_v29": getattr(args, "use_physical_space_temporal_loss_v29", False),
        "v29_floor_loss_weight": getattr(args, "v29_floor_loss_weight", 0.01),
        "v29_bone_temporal_weight": getattr(args, "v29_bone_temporal_weight", 0.01),
        "v29_com_jitter_weight": getattr(args, "v29_com_jitter_weight", 0.001),
        "v29_physical_loss_warmup_epochs": getattr(args, "v29_physical_loss_warmup_epochs", 0),
        # v31 physical collision penalty toggles
        "use_physical_collision_penalty_v31": getattr(args, "use_physical_collision_penalty_v31", False),
        "v31_collision_loss_weight": getattr(args, "v31_collision_loss_weight", 0.001),
        "v31_collision_margin": getattr(args, "v31_collision_margin", 0.05),
        "v31_collision_warmup_epochs": getattr(args, "v31_collision_warmup_epochs", 0),
        # v7 toggles
        "use_full_precision_dlt": args.use_full_precision_dlt,
        "use_robust_dlt_reweight": args.use_robust_dlt_reweight,
        "use_domain_embedding": args.use_domain_embedding,
        "num_domains": args.num_domains,
        # v11 toggles
        "use_irls_reweight": getattr(args, "use_irls_reweight", False),
        "irls_n_iters": getattr(args, "irls_n_iters", 2),
        "irls_cauchy_scale": getattr(args, "irls_cauchy_scale", 1.0),
    }
    if args.adaptive_view_k is not None:
        model_kwargs["adaptive_view_target_k"] = args.adaptive_view_k

    # v47 temporal aggregation kwargs are only passed when the flag is enabled so
    # the trainer can be imported/run on checkouts where the model has not yet
    # been wired for v47.
    if getattr(args, "use_v47_temporal_aggregation", False):
        model_kwargs.update(
            {
                "use_v47_temporal_aggregation": True,
                "v47_temporal_d_model": getattr(args, "v47_temporal_d_model", 64),
                "v47_temporal_n_heads": getattr(args, "v47_temporal_n_heads", 4),
                "v47_temporal_num_layers": getattr(args, "v47_temporal_num_layers", 2),
                "v47_temporal_window": getattr(args, "v47_temporal_window", None),
                "v47_temporal_dropout": getattr(args, "v47_temporal_dropout", 0.1),
                "v47_temporal_use_view_count_conditioning": getattr(
                    args, "v47_use_view_count_conditioning", True
                ),
            }
        )

    # v49-Lite temporal aggregation kwargs.
    if getattr(args, "use_v49_lite_temporal_aggregation", False):
        model_kwargs.update(
            {
                "use_v49_lite_temporal_aggregation": True,
                "v49_lite_temporal_d_model": getattr(args, "v49_lite_temporal_d_model", 32),
                "v49_lite_temporal_num_layers": getattr(args, "v49_lite_temporal_num_layers", 2),
                "v49_lite_temporal_kernel_size": getattr(args, "v49_lite_temporal_kernel_size", 3),
                "v49_lite_temporal_dropout": getattr(args, "v49_lite_temporal_dropout", 0.1),
                "v49_lite_temporal_residual_gate_init": getattr(args, "v49_lite_temporal_residual_gate_init", 0.0),
                "v49_lite_temporal_use_view_count_conditioning": getattr(
                    args, "v49_lite_temporal_use_view_count_conditioning", True
                ),
            }
        )

    # v51 domain-agnostic ensemble kwargs.
    if getattr(args, "use_v51_domain_agnostic_ensemble", False):
        model_kwargs.update(
            {
                "use_v51_domain_agnostic_ensemble": True,
                "v51_dae_hidden": getattr(args, "v51_dae_hidden", 64),
                "v51_dae_num_layers": getattr(args, "v51_dae_num_layers", 2),
                "v51_dae_dropout": getattr(args, "v51_dae_dropout", 0.1),
                "v51_dae_n_experts": getattr(args, "v51_dae_n_experts", 2),
                "v51_dae_identity_bypass": getattr(args, "v51_dae_identity_bypass", True),
                "v51_dae_min_weight": getattr(args, "v51_dae_min_weight", 0.05),
                "v51_dae_loss_weight": getattr(args, "v51_dae_loss_weight", 0.005),
            }
        )

    # v51 test-time self-evolution refiner kwargs.
    if getattr(args, "use_v51_test_time_self_evolution_refiner", False):
        model_kwargs.update(
            {
                "use_v51_test_time_self_evolution_refiner": True,
                "v51_tta_num_steps": getattr(args, "v51_tta_num_steps", 3),
                "v51_tta_lr": getattr(args, "v51_tta_lr", 1e-3),
                "v51_tta_hidden": getattr(args, "v51_tta_hidden", 32),
                "v51_tta_reproj_weight": getattr(args, "v51_tta_reproj_weight", 1.0),
                "v51_tta_temporal_weight": getattr(args, "v51_tta_temporal_weight", 0.5),
                "v51_tta_bone_weight": getattr(args, "v51_tta_bone_weight", 0.1),
                "v51_tta_entropy_weight": getattr(args, "v51_tta_entropy_weight", 0.01),
                "v51_tta_min_view_rel": getattr(args, "v51_tta_min_view_rel", 0.05),
                "v51_tta_max_view_rel": getattr(args, "v51_tta_max_view_rel", 1.0),
                "v51_tta_use_sefh_init": getattr(args, "v51_tta_use_sefh_init", True),
            }
        )

    # v50 self-evolution feedback head kwargs.
    if getattr(args, "use_v50_self_evolution_feedback_head", False):
        model_kwargs.update(
            {
                "use_v50_self_evolution_feedback_head": True,
                "v50_sefh_hidden": getattr(args, "v50_sefh_hidden", 64),
                "v50_sefh_num_layers": getattr(args, "v50_sefh_num_layers", 2),
                "v50_sefh_dropout": getattr(args, "v50_sefh_dropout", 0.1),
                "v50_sefh_reproj_weight": getattr(args, "v50_sefh_reproj_weight", 1.0),
                "v50_sefh_temporal_weight": getattr(args, "v50_sefh_temporal_weight", 0.5),
                "v50_sefh_epipolar_weight": getattr(args, "v50_sefh_epipolar_weight", 0.5),
                "v50_sefh_identity_init_gate": getattr(args, "v50_sefh_identity_init_gate", True),
            }
        )

    # v51 cross-domain sparse-view reliability kwargs.
    if getattr(args, "use_v51_cross_domain_sparse_view_reliability", False):
        model_kwargs.update(
            {
                "use_v51_cross_domain_sparse_view_reliability": True,
                "v51_cdsvr_hidden": getattr(args, "v51_cdsvr_hidden", 64),
                "v51_cdsvr_num_heads": getattr(args, "v51_cdsvr_num_heads", 4),
                "v51_cdsvr_dropout": getattr(args, "v51_cdsvr_dropout", 0.1),
                "v51_cdsvr_offset_min": getattr(args, "v51_cdsvr_offset_min", 0.05),
                "v51_cdsvr_use_domain_label": getattr(args, "v51_cdsvr_use_domain_label", True),
                "v51_cdsvr_uncertainty_temperature": getattr(args, "v51_cdsvr_uncertainty_temperature", 1.0),
                "v51_cdsvr_identity_init_gate": getattr(args, "v51_cdsvr_identity_init_gate", True),
            }
        )

    # v52 uncertainty-weighted triangulation kwargs.
    if getattr(args, "use_v52_uncertainty_weighted_triangulation", False):
        model_kwargs.update(
            {
                "use_v52_uncertainty_weighted_triangulation": True,
                "v52_uwt_hidden": getattr(args, "v52_uwt_hidden", 64),
                "v52_uwt_n_layers": getattr(args, "v52_uwt_n_layers", 2),
                "v52_uwt_weight_type": getattr(args, "v52_uwt_weight_type", "per_view_joint"),
                "v52_uwt_temperature": getattr(args, "v52_uwt_temperature", 1.0),
                "v52_uwt_use_geometry_bias": getattr(args, "v52_uwt_use_geometry_bias", True),
                "v52_uwt_use_feature_bias": getattr(args, "v52_uwt_use_feature_bias", True),
                "v52_uwt_identity_init": getattr(args, "v52_uwt_identity_init", True),
                "v52_uwt_min_weight": getattr(args, "v52_uwt_min_weight", 0.05),
                "v52_uwt_loss_weight": getattr(args, "v52_uwt_loss_weight", 0.01),
                "v52_uwt_damping": getattr(args, "v52_uwt_damping", 1e-4),
                "v52_uwt_warmup_epochs": getattr(args, "v52_uwt_warmup_epochs", 0),
            }
        )

    # v53 physical-space calibration kwargs.
    if getattr(args, "use_v53_physical_space_calibration", False):
        model_kwargs.update(
            {
                "use_v53_physical_space_calibration": True,
                "v53_psc_hidden": getattr(args, "v53_psc_hidden", 64),
                "v53_psc_n_layers": getattr(args, "v53_psc_n_layers", 2),
                "v53_psc_identity_init": getattr(args, "v53_psc_identity_init", True),
                "v53_psc_residual_gate_init": getattr(args, "v53_psc_residual_gate_init", -6.0),
                "v53_psc_use_uwt_weights": getattr(args, "v53_psc_use_uwt_weights", True),
                "v53_psc_use_floor": getattr(args, "v53_psc_use_floor", True),
                "v53_psc_use_bone_scale": getattr(args, "v53_psc_use_bone_scale", True),
                "v53_psc_loss_weight": getattr(args, "v53_psc_loss_weight", 1.0),
                "v53_psc_floor_weight": getattr(args, "v53_psc_floor_weight", 0.01),
                "v53_psc_bone_weight": getattr(args, "v53_psc_bone_weight", 0.1),
                "v53_psc_reproj_weight": getattr(args, "v53_psc_reproj_weight", 0.1),
                "v53_psc_warmup_epochs": getattr(args, "v53_psc_warmup_epochs", 0),
                "v53_psc_min_visible_views": getattr(args, "v53_psc_min_visible_views", 2),
            }
        )

    # v54 physical-space calibration v2 kwargs.
    if getattr(args, "use_v54_physical_space_calibration_v2", False):
        model_kwargs.update(
            {
                "use_v54_physical_space_calibration_v2": True,
                "v54_psc2_hidden": getattr(args, "v54_psc2_hidden", 64),
                "v54_psc2_n_layers": getattr(args, "v54_psc2_n_layers", 2),
                "v54_psc2_num_domains": getattr(args, "v54_psc2_num_domains", 8),
                "v54_psc2_use_floor": getattr(args, "v54_psc2_use_floor", True),
                "v54_psc2_use_contact": getattr(args, "v54_psc2_use_contact", True),
                "v54_psc2_use_bone_scale": getattr(args, "v54_psc2_use_bone_scale", True),
                "v54_psc2_use_temporal_smoothness": getattr(args, "v54_psc2_use_temporal_smoothness", True),
                "v54_psc2_use_gnn": getattr(args, "v54_psc2_use_gnn", True),
                "v54_psc2_gnn_layers": getattr(args, "v54_psc2_gnn_layers", 1),
                "v54_psc2_identity_init": getattr(args, "v54_psc2_identity_init", True),
                "v54_psc2_residual_gate_init": getattr(args, "v54_psc2_residual_gate_init", -6.0),
                "v54_psc2_loss_weight": getattr(args, "v54_psc2_loss_weight", 1.0),
                "v54_psc2_floor_weight": getattr(args, "v54_psc2_floor_weight", 0.01),
                "v54_psc2_bone_weight": getattr(args, "v54_psc2_bone_weight", 0.05),
                "v54_psc2_contact_weight": getattr(args, "v54_psc2_contact_weight", 0.01),
                "v54_psc2_temporal_weight": getattr(args, "v54_psc2_temporal_weight", 0.01),
                "v54_psc2_reproj_weight": getattr(args, "v54_psc2_reproj_weight", 0.1),
                "v54_psc2_contact_velocity_thresh": getattr(args, "v54_psc2_contact_velocity_thresh", 0.3),
                "v54_psc2_min_visible_views": getattr(args, "v54_psc2_min_visible_views", 2),
                "v54_psc2_warmup_epochs": getattr(args, "v54_psc2_warmup_epochs", 0),
            }
        )

    # v48 domain generalization kwargs are only passed when the flag is enabled
    # so the trainer can be imported/run on checkouts where the model has not
    # yet been wired for v48.
    if getattr(args, "use_v48_domain_generalization", False):
        v48_num_domains = max(getattr(args, "num_domains", 2), 6)
        model_kwargs.update(
            {
                "use_v48_domain_generalization": True,
                "v48_dg_hidden": getattr(args, "v48_dg_hidden", 64),
                "v48_dg_num_domains": v48_num_domains,
                "v48_dg_dropout": 0.1,
                "v48_dg_use_domain_film": getattr(args, "v48_dg_use_domain_film", True),
                "v48_dg_use_grl_discriminator": True,
                "v48_dg_grl_lambda": getattr(args, "v48_dg_grl_lambda", 0.1),
            }
        )

    model = OmniMultiViewFusionV5(**model_kwargs)

    if n_joints != 17 and n_joints == 28 and hasattr(model, "rebuild_graph"):
        model.rebuild_graph(n_joints, dataset="mpiinf3dhp")

    if device is not None:
        model = model.to(device)

    return model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parent_indices(j: int) -> List[int]:
    """Return the parent list for the active skeleton."""
    if j == 17:
        return H36M_17_PARENTS
    if j == 28:
        return MPI_INF_3DHP_28_PARENTS
    raise ValueError(f"No built-in parent list for j={j}")


# ---------------------------------------------------------------------------
# Data loading (same pattern as train_omniview_fusion_v4_webbridge_multi.py)
# ---------------------------------------------------------------------------

class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a sequence; useful for train set augmentation."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


# ---------------------------------------------------------------------------
# Synthetic smoke dataset (CPU-friendly, no external data required)
# ---------------------------------------------------------------------------

def _make_synthetic_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a circular rig of pinhole cameras and return K, R, t tensors."""
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    K = torch.from_numpy(np.stack(K_list, axis=0)).float()
    R = torch.from_numpy(np.stack(R_list, axis=0)).float()
    t = torch.from_numpy(np.stack(t_list, axis=0)).float()
    return K, R, t


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 100,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = self._project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - self.clip_len + 1)

    @staticmethod
    def _project_points(
        joints_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Project world points into all views."""
        X = joints_3d  # (F, J, 3)
        t = t[:, None, None, :]
        X_cam = torch.einsum("vab,fjb->vfja", R, X) + t  # (V, F, J, 3)
        z = X_cam[..., 2:3].clamp(min=1e-6)
        uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])  # (V, F, J, 3, 1)
        points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
        return points_2d.permute(1, 0, 2, 3)  # (F, V, J, 2)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int):
        start = idx
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def inject_outlier_views(
    x: torch.Tensor,
    prob: float = 0.0,
    max_views: int = 1,
    offset_std: float = 10.0,
    noise_std: float = 15.0,
    min_views: int = 2,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Corrupt a random subset of views with large per-view offsets and pixel noise.

    Args:
        x: (B, T, V, J, 3) tensor of (x, y, confidence) observations.
        prob: Per-clip probability of corrupting any views.
        max_views: Maximum number of views to corrupt in a single clip.
        offset_std: Standard deviation of the large per-view 2-D offset (pixels).
        noise_std: Standard deviation of the per-pixel Gaussian noise (pixels).
        min_views: Ensure at least this many views remain uncorrupted.

    Returns:
        A tuple of (corrupted x, outlier_labels). outlier_labels is None when no
        corruption is applied or when prob <= 0, otherwise a (B, T, V, J) float
        tensor with 1.0 for corrupted (view, joint) entries and 0.0 otherwise.
    """
    if prob <= 0.0 or max_views <= 0:
        return x, None

    # Work on a clone so in-place indexing does not interfere with gradients
    # on the tensor produced by ``augment_clip``.
    x = x.clone()
    B, T, V, J, _ = x.shape
    outlier_labels = torch.zeros(B, T, V, J, device=x.device, dtype=x.dtype)
    for i in range(B):
        if torch.rand(1, device=x.device).item() >= prob:
            continue
        k_max = min(max_views, max(0, V - min_views))
        if k_max <= 0:
            continue
        k = torch.randint(1, k_max + 1, (1,)).item()
        idx = torch.randperm(V, device=x.device)[:k]
        if idx.numel() == 0:
            continue
        # Per-view large offset shared across the whole clip.
        offset = torch.randn(1, idx.numel(), 1, 2, device=x.device, dtype=x.dtype) * offset_std
        # Per-pixel Gaussian noise.
        noise = torch.randn(T, idx.numel(), J, 2, device=x.device, dtype=x.dtype) * noise_std
        x[i, :, idx, :, :2] = x[i, :, idx, :, :2] + offset + noise
        # Confidence remains > 0 so the robust DLT path must down-weight the view.
        outlier_labels[i, :, idx, :] = 1.0
    return x, outlier_labels


def augment_occlusion_noise(
    x: torch.Tensor,
    prob: float = 0.0,
    joint_occlusion_rate: float = 0.1,
    multiview_occlusion_rate: float = 0.05,
    temporal_occlusion_prob: float = 0.1,
    per_joint_noise_std: float = 1.5,
    per_view_noise_std: float = 0.5,
    min_visible_joints: int = 5,
) -> torch.Tensor:
    """Joint-level occlusion and structured noise augmentation.

    Simulates three failure modes beyond whole-view outliers:

    1. Per-joint occlusion in a single view (confidence -> 0).
    2. Multi-view joint occlusion: the same joint is occluded in several
       correlated views.
    3. Temporal joint occlusion: a joint is occluded across consecutive frames.

    Also adds localized Gaussian noise to 2-D keypoints.

    Args:
        x: (B, T, V, J, 3) tensor of (x, y, confidence) observations.
        prob: Per-clip probability of applying any augmentation in this family.
        joint_occlusion_rate: Probability of occluding each (view, joint) pair.
        multiview_occlusion_rate: Fraction of joints to occlude across multiple views.
        temporal_occlusion_prob: Probability of applying a temporal occlusion streak.
        per_joint_noise_std: Std of Gaussian noise added independently per joint.
        per_view_noise_std: Std of shared Gaussian noise added per view.
        min_visible_joints: Ensure at least this many joints remain visible per view.

    Returns:
        Augmented tensor (same shape as input).
    """
    if prob <= 0.0:
        return x

    x = x.clone()
    B, T, V, J, _ = x.shape

    for i in range(B):
        if torch.rand(1, device=x.device).item() >= prob:
            continue

        # Per-joint single-view occlusions.
        if joint_occlusion_rate > 0.0:
            occ_mask = torch.rand(T, V, J, device=x.device) < joint_occlusion_rate
            x[i, :, :, :, 2] = x[i, :, :, :, 2] * (~occ_mask).float()

        # Multi-view joint occlusions.
        if multiview_occlusion_rate > 0.0:
            n_joints_to_occlude = max(1, int(multiview_occlusion_rate * J))
            if n_joints_to_occlude > 0 and V > 1:
                j_idx = torch.randperm(J, device=x.device)[:n_joints_to_occlude]
                k_views = torch.randint(1, V, (1,)).item()
                v_idx = torch.randperm(V, device=x.device)[:k_views]
                x[i, :, v_idx[:, None], j_idx[None, :], 2] = 0.0

        # Temporal joint occlusions.
        if temporal_occlusion_prob > 0.0 and T > 2:
            if torch.rand(1, device=x.device).item() < temporal_occlusion_prob:
                j_idx = torch.randint(0, J, (1,), device=x.device).item()
                streak_len = max(1, min(T, torch.randint(T // 4, T // 2 + 1, (1,)).item()))
                start = torch.randint(0, max(1, T - streak_len + 1), (1,)).item()
                x[i, start : start + streak_len, :, j_idx, 2] = 0.0

        # Structured noise.
        if per_joint_noise_std > 0.0:
            noise_joint = torch.randn(T, V, J, 2, device=x.device, dtype=x.dtype) * per_joint_noise_std
            x[i, :, :, :, :2] = x[i, :, :, :, :2] + noise_joint

        if per_view_noise_std > 0.0:
            noise_view = torch.randn(T, V, 1, 2, device=x.device, dtype=x.dtype) * per_view_noise_std
            x[i, :, :, :, :2] = x[i, :, :, :, :2] + noise_view

        # Safety: keep a minimum number of visible joints per view.
        visible_per_view = (x[i, :, :, :, 2] > 0).float().sum(dim=-1)  # (T, V)
        if visible_per_view.min().item() < min_visible_joints:
            offending = visible_per_view < min_visible_joints  # (T, V)
            for t, v in offending.nonzero(as_tuple=False):
                x[i, t, v, :min_visible_joints, 2] = x[i, t, v, :min_visible_joints, 2].clamp(min=1e-3)

    return x


def augment_clip(
    x: torch.Tensor,
    *,
    noise_std: float = 0.5,
    dropout_rate: float = 0.0,
    view_dropout_rate: float = 0.0,
    min_views: int = 2,
    variable_view_subset: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Lightweight per-clip augmentation with optional view dropout.

    When ``variable_view_subset`` is True, instead of independent per-view
    dropout we sample a random subset size ``k ~ Uniform(min_views, V)``
    and keep exactly ``k`` views per clip. This forces the model to learn
    from arbitrary view cardinalities.
    """
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask

    view_mask = None
    if variable_view_subset:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = torch.zeros(B, V, device=x.device)
        for i in range(B):
            k = torch.randint(min_views, V + 1, (1,)).item()
            idx = torch.randperm(V)[:k]
            view_mask[i, idx] = 1.0
        x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)
    elif view_dropout_rate > 0:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = (torch.rand(B, V, device=x.device) > view_dropout_rate).float()
        for i in range(B):
            kept = view_mask[i].nonzero(as_tuple=True)[0]
            if kept.numel() < min_views:
                dropped = (view_mask[i] == 0).nonzero(as_tuple=True)[0]
                needed = min_views - kept.numel()
                if needed > 0 and dropped.numel() > 0:
                    perm = torch.randperm(dropped.numel())
                    extra = dropped[perm[:needed]]
                    view_mask[i, extra] = 1.0
        x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)

    return x, view_mask


def apply_per_domain_view_dropout(
    x: torch.Tensor,
    dataset_id: torch.Tensor,
    dropout_per_domain: Dict[str, float],
    min_views: int = 2,
) -> torch.Tensor:
    """Apply per-sample view dropout with a probability that depends on domain.

    Args:
        x: (B, T, V, C) tensor of observations; confidence lives in channel 2.
        dataset_id: (B,) or (B, 1) integer domain ids.
        dropout_per_domain: Mapping from domain id string to dropout probability.
        min_views: Minimum number of views to retain per sample.

    Returns:
        x with confidence channel zeroed for dropped views.
    """
    B, T, V, C = x.shape
    device = x.device
    dataset_id = dataset_id.squeeze(-1).long()
    for i in range(B):
        d = str(dataset_id[i].item())
        p = float(dropout_per_domain.get(d, 0.0))
        if p <= 0.0:
            continue
        mask = (torch.rand(V, device=device) > p).float()
        if int(mask.sum().item()) < min_views:
            needed = min_views - int(mask.sum().item())
            dropped = (mask == 0).nonzero(as_tuple=True)[0]
            if dropped.numel() > 0:
                perm = torch.randperm(dropped.numel())
                extra = dropped[perm[:needed]]
                mask[extra] = 1.0
        x[i, :, :, 2] = x[i, :, :, 2] * mask.view(1, V, 1)
    return x


# ---------------------------------------------------------------------------
# Calibration perturbation helpers
# ---------------------------------------------------------------------------

def _camera_perturbation_schedule(epoch: int, args: Namespace) -> Dict[str, float]:
    """Return the per-epoch camera perturbation standard deviations."""
    try:
        from motionflow_mv.calibration.camera_perturbation_curriculum import (  # noqa: E402
            extended_camera_perturbation_schedule_with_anneal,
        )

        return extended_camera_perturbation_schedule_with_anneal(
            epoch,
            total_epochs=args.epochs,
            schedule=args.cam_aug_schedule,
            rot=args.cam_aug_rot,
            trans=args.cam_aug_trans,
            focal=args.cam_aug_focal,
            pp=args.cam_aug_pp,
            ramp_epochs=args.cam_aug_ramp_epochs,
            intrinsics_ramp_epochs=args.cam_aug_intrinsics_ramp_epochs,
            warmup_epochs=args.cam_aug_warmup_epochs,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Camera curriculum unavailable ({exc}); using flat perturbations.")
        return {
            "rot_std": args.cam_aug_rot,
            "trans_std": args.cam_aug_trans,
            "focal_std": args.cam_aug_focal,
            "pp_std": args.cam_aug_pp,
        }


def apply_calibration_perturbation(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    epoch: int,
    args: Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply epoch-dependent calibration perturbations to cameras."""
    if args.cam_aug_schedule == "none":
        return K, R, t
    stds = _camera_perturbation_schedule(epoch, args)
    return perturb_cameras(
        K,
        R,
        t,
        rot_std=stds["rot_std"],
        trans_std=stds["trans_std"],
        focal_std=stds["focal_std"],
        pp_std=stds["pp_std"],
    )


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def project_points_3d_to_2d(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project 3-D points into each view.

    Args:
        points_3d: (B, T, J, 3)
        K, R, t: (B, V, 3, 3) / (B, V, 3)

    Returns:
        uv: (B, T, V, J, 2)
    """
    B, T, J, _ = points_3d.shape
    V = K.shape[1]
    K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
    R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
    t = t.unsqueeze(1).expand(-1, T, -1, -1)
    X_cam = torch.einsum("btvac,btjc->btvja", R, points_3d) + t.unsqueeze(3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv_h = torch.einsum("btvik,btvjk->btvji", K, X_cam / z)
    return uv_h[..., :2]


def uncertainty_nll_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    confidences: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of 2-D reprojection residuals under L L^T.

    Computes the residual between predicted and observed 2-D keypoints, solves
    ``L y = r`` for the Mahalanobis distance, and regularises the diagonals of
    ``L`` to avoid singularities.  The loss is averaged over valid (visible)
    joints and safely handles an empty valid mask.
    """
    uv_pred = project_points_3d_to_2d(pred_3d, K, R, t)  # (B, T, V, J, 2)
    r = uv_pred - points_2d  # (B, T, V, J, 2)
    valid = confidences > 0  # (B, T, V, J)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)

    r_flat = r.reshape(-1, 2)[valid.reshape(-1)]
    L_flat = L.reshape(-1, 2, 2)[valid.reshape(-1)]

    # Clamp diagonals to keep L invertible (out-of-place to keep autograd happy).
    eye2 = torch.eye(2, device=L_flat.device, dtype=torch.bool)
    diag = L_flat.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
    L_flat = L_flat * (~eye2) + eye2 * diag.unsqueeze(-2)

    # Solve L y = r and use ||y||^2 as the Mahalanobis distance.
    y = torch.linalg.solve(L_flat, r_flat.unsqueeze(-1)).squeeze(-1)
    mahalanobis = (y ** 2).sum(dim=-1)
    mahalanobis = mahalanobis.clamp(0.0, 50.0)

    log_det = 2.0 * (torch.log(L_flat[..., 0, 0]) + torch.log(L_flat[..., 1, 1]))

    nll = 0.5 * (mahalanobis + log_det + 2.0 * math.log(2.0 * math.pi))
    return nll.mean()


def bone_length_loss(pred_3d: torch.Tensor, target_3d: torch.Tensor, parents: List[int]) -> torch.Tensor:
    """MSE between predicted and target bone vectors."""
    loss = torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)
    count = 0
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        pred_bone = pred_3d[..., child, :] - pred_3d[..., parent, :]
        tgt_bone = target_3d[..., child, :] - target_3d[..., parent, :]
        loss = loss + F.mse_loss(pred_bone, tgt_bone)
        count += 1
    return loss / max(1, count)


def attention_entropy_loss(weights: torch.Tensor, dim: int = -3) -> torch.Tensor:
    """Entropy regularisation on per-view triangulation weights."""
    p = weights / (weights.sum(dim=dim, keepdim=True) + 1e-8)
    p = p.clamp(min=1e-8)
    entropy = -(p * torch.log(p)).sum(dim=dim).mean()
    return entropy


def budget_loss(weights: torch.Tensor, target_k: float) -> torch.Tensor:
    """Mean-squared deviation between the number of active views and ``target_k``."""
    active = (weights > 0.1).float().sum(dim=-2).mean()
    return F.mse_loss(active, torch.tensor(target_k, device=weights.device, dtype=weights.dtype))


def _reprojection_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    view_mask: torch.Tensor,
) -> torch.Tensor:
    """Robust, focal-normalized 2D reprojection loss for the final 3D prediction.

    Args:
        pred_3d: (B, T, J, 3)
        points_2d: (B, T, V, J, 2)
        K: (B, T, V, 3, 3)
        R: (B, T, V, 3, 3)
        t: (B, T, V, 3)
        view_mask: (B, T, V)

    Returns:
        Scalar robust reprojection error in focal-length units.
    """
    B, T, V, J = points_2d.shape[0], points_2d.shape[1], points_2d.shape[2], points_2d.shape[3]
    if K.dim() == 4:
        K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
        R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
        t = t.unsqueeze(1).expand(-1, T, -1, -1)
    X_h = torch.cat([pred_3d, torch.ones(B, T, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)], dim=-1)
    X_h = X_h.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B,T,V,J,4)
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    P = (K @ Rt).unsqueeze(3).expand(-1, -1, -1, J, -1, -1)  # (B,T,V,J,3,4)
    x_h = (P @ X_h.unsqueeze(-1)).squeeze(-1)
    x_pred = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
    diff = x_pred - points_2d

    # Normalize by focal length so reprojection error is in focal-length units
    # and on a comparable scale to the 3-D MSE loss (meters^2).
    f = (K[..., 0, 0] + K[..., 1, 1]) / 2.0  # (B, T, V)
    f = f.unsqueeze(-1).unsqueeze(-1)  # (B, T, V, 1, 1)
    diff = diff / (f + 1e-6)

    # Charbonnier / pseudo-Huber loss for robustness to outliers.
    rho = torch.sqrt(diff ** 2 + 1e-4 ** 2)

    mask = view_mask.unsqueeze(-1).unsqueeze(-1)
    return (rho * mask).sum() / (mask.sum() + 1e-8)


def temporal_consistency_loss(
    pred_3d: torch.Tensor,
    y: torch.Tensor,
    temporal_loss_weight: float,
    temporal_acceleration_weight: float,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Velocity + acceleration temporal consistency loss.

    Parameters
    ----------
    pred_3d:
        Predicted 3-D poses, shape ``(B, T, J, 3)``.
    y:
        Ground-truth 3-D pose, shape ``(B, T, J, 3)``.
    temporal_loss_weight:
        Overall multiplier for the temporal loss.
    temporal_acceleration_weight:
        Relative weight of the acceleration smoothness term (0 disables).
    eps:
        Charbonnier epsilon.

    Returns
    -------
    total_loss:
        Scalar loss already scaled by ``temporal_loss_weight``.
    metrics:
        Dictionary with ``temp_vel_loss``, ``temp_acc_loss`` and ``temp_loss``.
    """
    if temporal_loss_weight <= 0.0 or pred_3d.shape[1] < 2:
        return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype), {}

    diff = pred_3d - y
    vel = diff[:, 1:] - diff[:, :-1]
    vel_loss = torch.sqrt(vel.pow(2).sum(-1) + eps ** 2).mean()

    acc_loss = torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)
    if temporal_acceleration_weight > 0.0 and pred_3d.shape[1] > 2:
        acc = diff[:, 2:] - 2 * diff[:, 1:-1] + diff[:, :-2]
        acc_loss = torch.sqrt(acc.pow(2).sum(-1) + eps ** 2).mean()

    total = vel_loss + temporal_acceleration_weight * acc_loss
    return temporal_loss_weight * total, {
        "temp_vel_loss": vel_loss.item(),
        "temp_acc_loss": acc_loss.item() if temporal_acceleration_weight > 0.0 else 0.0,
        "temp_loss": total.item(),
    }


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------

def _skeleton_config_for_joints(j: int):
    """Return (parents, symmetry_pairs, foot_indices) for known skeletons."""
    if j == 17:
        return H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, [11, 14]
    if j == 28:
        return MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS, [8, 10]
    return None, None, None


def build_compute_loss(args: Namespace):
    """Build the compute_loss closure used by TrainerV2."""
    parents = None
    if (
        args.bone_loss_weight > 0.0
        or args.joint_limit_weight > 0.0
        or args.temporal_bone_weight > 0.0
    ):
        try:
            parents = get_parent_indices(args.j)
        except ValueError:
            parents = None

    # Optional v40 skeleton-aware physical loss.
    skeleton_physical_loss = None
    if getattr(args, "use_skeleton_physical_loss_v40", False):
        sk_parents, sk_sym, sk_feet = _skeleton_config_for_joints(args.j)
        if sk_parents is not None:
            skeleton_physical_loss = SkeletonPhysicalLossV40(
                parents=sk_parents,
                symmetry_pairs=sk_sym,
                foot_indices=sk_feet,
                bone_weight=args.v40_bone_weight,
                joint_limit_weight=args.v40_joint_limit_weight,
                symmetry_weight=args.v40_symmetry_weight,
                floor_weight=args.v40_floor_weight,
                warmup_epochs=args.v40_warmup_epochs,
            )

    # Optional v41 per-domain loss weights for mixed-loader training.
    domain_loss_weights: Optional[torch.Tensor] = None
    if getattr(args, "domain_loss_weights", None) is not None:
        weights = [float(w.strip()) for w in args.domain_loss_weights.split(",")]
        domain_loss_weights = torch.tensor(weights, dtype=torch.float32)

    # v48 domain-difficulty-weighted loss (DDWL) state.  Maintains a running
    # EMA of per-domain MSE; weights are inverse to the EMA so harder domains
    # receive larger gradients.  The state lives in the closure so it persists
    # across training steps.
    use_v48_dg = getattr(args, "use_v48_domain_generalization", False)
    v48_use_ddwl = use_v48_dg and getattr(args, "v48_dg_use_ddwl", True)
    v48_ddwl_temperature = getattr(args, "v48_dg_ddwl_temperature", 2.0)
    v48_ddwl_warmup_epochs = getattr(args, "v48_dg_ddwl_warmup_epochs", 1)
    v48_ddwl_beta = 0.9
    v48_num_domains_ddwl = max(getattr(args, "num_domains", 2), 6) if use_v48_dg else getattr(args, "num_domains", 2)
    v48_loss_ema: Dict[int, float] = {d: 0.0 for d in range(v48_num_domains_ddwl)}

    def compute_loss(
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Mixed loader returns (x, y, K, R, t, dataset_id); the dataset_id is used
        # to build a base view mask for padded views.
        if len(batch) == 6:
            x, y, K, R, t, dataset_id = batch
            dataset_id = dataset_id.to(device)
        else:
            x, y, K, R, t = batch
            dataset_id = None

        x = x.to(device)
        y = y.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        # Training-only augmentations.  Operate on a clone so the DataLoader
        # buffer is not silently modified.
        #
        # v46 sparse-view dropout: when enabled, ramp the per-view dropout
        # probability from 0 up to the target over the first half of training
        # if the curriculum flag is set.  This is applied via the existing
        # ``augment_clip`` view-dropout path so the model sees variable
        # cardinalities without extra data-loader changes.
        # v48 per-domain view dropout: when enabled, it takes precedence over
        # the global v46 view dropout probability.  The global v46 probability
        # is set to 0 here and the per-domain schedule is applied after
        # ``augment_clip`` so different domains can see different dropout rates.
        v48_dropout_per_domain: Optional[Dict[str, float]] = None
        if use_v48_dg and getattr(args, "v48_dropout_per_domain", None) is not None:
            v48_dropout_per_domain = args.v48_dropout_per_domain  # type: ignore[assignment]

        if getattr(args, "use_v46_sparse_view_generalization", False):
            if v48_dropout_per_domain is not None:
                v46_dropout_prob = 0.0
                v46_min_views = args.v46_svg_min_views
            else:
                v46_dropout_prob = args.v46_svg_view_dropout_prob
                if args.v46_svg_use_curriculum and args.epochs > 1:
                    ramp_epochs = max(1, args.epochs // 2)
                    current_epoch = getattr(model, "epoch", 1) - 1
                    progress = min(1.0, max(0.0, current_epoch / ramp_epochs))
                else:
                    progress = 1.0
                v46_dropout_prob = v46_dropout_prob * progress
                v46_min_views = args.v46_svg_min_views
        else:
            v46_dropout_prob = 0.0 if v48_dropout_per_domain is not None else args.view_dropout_rate
            v46_min_views = args.min_views

        x, view_mask = augment_clip(
            x.clone(),
            noise_std=args.noise_std if not args.smoke else 0.0,
            dropout_rate=args.confidence_dropout,
            view_dropout_rate=v46_dropout_prob,
            min_views=v46_min_views,
            variable_view_subset=args.variable_view_subset,
        )

        if v48_dropout_per_domain is not None and dataset_id is not None:
            x = apply_per_domain_view_dropout(
                x,
                dataset_id,
                v48_dropout_per_domain,
                min_views=v46_min_views,
            )

        # Optional outlier-view augmentation: corrupt a random subset of views with
        # large offsets and pixel noise.  Confidence remains > 0 so the robust DLT
        # path must learn to down-weight the bad views.  When available, the
        # augmentation mask is passed to the model for optional supervised training
        # of the v33 outlier-view detector.
        outlier_labels = None
        if args.outlier_view_prob > 0.0:
            x, outlier_labels = inject_outlier_views(
                x,
                prob=args.outlier_view_prob,
                max_views=args.outlier_view_max_views,
                offset_std=args.outlier_view_offset_std,
                noise_std=args.outlier_view_noise_std,
                min_views=args.min_views,
            )

        # Optional joint-level occlusion / structured noise augmentation.
        if getattr(args, "occlusion_augment_prob", 0.0) > 0.0:
            x = augment_occlusion_noise(
                x,
                prob=args.occlusion_augment_prob,
                joint_occlusion_rate=args.occlusion_joint_rate,
                multiview_occlusion_rate=args.occlusion_multiview_rate,
                temporal_occlusion_prob=args.occlusion_temporal_prob,
                per_joint_noise_std=args.occlusion_per_joint_noise_std,
                per_view_noise_std=args.occlusion_per_view_noise_std,
                min_visible_joints=args.occlusion_min_visible_joints,
            )

        # When using the mixed loader, padded views must be masked out.  The
        # dataset_id encodes the source domain (0=H36M/4 views, 1=MPI/14 views).
        if dataset_id is not None:
            B, T, V_full = x.shape[0], x.shape[1], x.shape[2]
            base_view_mask = torch.zeros(B, T, V_full, device=device)
            for i in range(B):
                if dataset_id[i].item() == 0:  # H36M
                    base_view_mask[i, :, :4] = 1.0
                elif dataset_id[i].item() == 1:  # MPI-INF-3DHP
                    base_view_mask[i, :, :14] = 1.0
                else:
                    base_view_mask[i, :, :].fill_(1.0)
            if view_mask is None:
                view_mask = base_view_mask
            else:
                # view_mask comes from augment_clip with shape (B, V); base_view_mask
                # is (B, T, V).  Unsqueeze to align the view dimension.
                if view_mask.dim() == 2:
                    view_mask = view_mask.unsqueeze(1) * base_view_mask
                else:
                    view_mask = view_mask * base_view_mask

        # Optional variable-view training: sample a random subset of views and
        # optionally permute the order of all views.  The model still receives
        # all V camera tensors (so internal buffers sized for ``n_views`` stay
        # valid), but ``view_mask`` masks out the unselected views, giving a
        # variable effective number of actual views per batch.
        if args.use_variable_view_training:
            B, T, V_full = x.shape[0], x.shape[1], x.shape[2]
            # Curriculum: gradually raise the maximum number of active views
            # over epochs. Early epochs see smaller view subsets (easier), later
            # epochs see the full target distribution.
            progress = min(1.0, (getattr(model, "epoch", 1) - 1) / max(1, args.epochs))
            k_max_start = args.variable_view_max_views_start
            if k_max_start is None:
                k_max_start = args.variable_view_max_views
            k_max_eff = (
                k_max_start
                + (args.variable_view_max_views - k_max_start)
                * (progress ** args.variable_view_curriculum_alpha)
            )
            k_max_eff = int(round(k_max_eff))
            k_max_eff = min(args.variable_view_max_views, max(k_max_start, k_max_eff))
            k_max_eff = max(args.variable_view_min_views, min(k_max_eff, V_full))
            # Domain-aware clamp: H36M has 4 real views, MPI has 14.  WebBridge
            # and other domains fall back to the full tensor width.
            domain_real_views = {0: 4, 1: 14}
            view_mask = torch.zeros(B, T, V_full, device=device)
            for i in range(B):
                real_v = V_full
                if args.domain_aware_view_curriculum and dataset_id is not None:
                    real_v = domain_real_views.get(dataset_id[i].item(), V_full)
                k_i = torch.randint(
                    args.variable_view_min_views,
                    min(k_max_eff, real_v) + 1,
                    (1,),
                ).item()
                selected = torch.randperm(V_full)[:k_i]
                view_mask[i, :, selected] = 1.0

            if args.variable_view_permute:
                perm = torch.randperm(V_full)
                x = x[:, :, perm, :, :]
                K = K[:, perm, :, :]
                R = R[:, perm, :, :]
                t = t[:, perm, :]
                view_mask = view_mask[:, :, perm]
                if outlier_labels is not None:
                    outlier_labels = outlier_labels[:, :, perm, :]

        # Optional calibration curriculum.
        K_aug, R_aug, t_aug = apply_calibration_perturbation(
            K, R, t, model.epoch if hasattr(model, "epoch") else 0, args
        )

        model_kwargs_forward: Dict[str, Any] = {}
        if view_mask is not None:
            model_kwargs_forward["view_mask"] = view_mask
        if (
            (args.use_domain_embedding or use_v48_dg or args.use_v52_uncertainty_weighted_triangulation or args.use_v53_physical_space_calibration or args.use_v54_physical_space_calibration_v2)
            and dataset_id is not None
        ):
            model_kwargs_forward["domain_id"] = dataset_id
        if outlier_labels is not None:
            model_kwargs_forward["outlier_labels"] = outlier_labels
        out = model(x, K=K_aug, R=R_aug, t=t_aug, **model_kwargs_forward)
        pred_3d = out[0]
        weights = out[1]
        visibility = out[2]
        L = out[3]
        epi_loss = out[4]
        entropy_loss_out = out[5] if len(out) > 5 else None
        budget_loss_out = out[6] if len(out) > 6 else None
        # v48 domain adapter may return an adversarial / gradient-reversal loss.
        v48_domain_loss = out[7] if len(out) > 7 else None

        # v41 static or v48 dynamic per-domain loss weighting for mixed training.
        mse_per_sample = F.mse_loss(pred_3d, y, reduction="none").mean(dim=(1, 2, 3))
        sample_weights: Optional[torch.Tensor] = None
        if dataset_id is not None and v48_use_ddwl:
            # Update per-domain EMA with the detached current batch loss.
            with torch.no_grad():
                for d in range(v48_num_domains_ddwl):
                    mask = (dataset_id.squeeze(-1) == d)
                    if mask.any():
                        mean_loss_d = mse_per_sample[mask].mean().item()
                        v48_loss_ema[d] = (
                            v48_ddwl_beta * v48_loss_ema[d]
                            + (1.0 - v48_ddwl_beta) * mean_loss_d
                        )
            # After warmup, give harder domains higher weight.
            current_epoch = getattr(model, "epoch", 1) - 1
            if current_epoch >= v48_ddwl_warmup_epochs:
                ema_t = torch.tensor(
                    [max(v48_loss_ema[d], 1e-8) for d in range(v48_num_domains_ddwl)],
                    device=device,
                    dtype=torch.float32,
                )
                w_d_raw = (ema_t / ema_t.max()) ** (-1.0 / v48_ddwl_temperature)
                w_d = w_d_raw / w_d_raw.sum() * v48_num_domains_ddwl
                w_d = w_d.clamp(0.5, 2.0)
            else:
                w_d = torch.ones(v48_num_domains_ddwl, device=device, dtype=torch.float32)
            sample_weights = w_d[dataset_id.squeeze(-1).long()]
            loss = (mse_per_sample * sample_weights).mean()
            metrics: Dict[str, Any] = {
                "mpjpe": (pred_3d - y).norm(dim=-1).mean().item(),
                "v48_ddwl_weights": w_d.detach().cpu().numpy().tolist(),
            }
        elif dataset_id is not None and domain_loss_weights is not None:
            domain_sample_weights = domain_loss_weights.to(device)[dataset_id.squeeze(-1).long()]
            loss = (mse_per_sample * domain_sample_weights).mean()
            metrics = {
                "mpjpe": (pred_3d - y).norm(dim=-1).mean().item(),
            }
        else:
            loss = mse_per_sample.mean()
            metrics = {
                "mpjpe": (pred_3d - y).norm(dim=-1).mean().item(),
            }

        if epi_loss is not None:
            loss = loss + epi_loss

        # Optional v52 uncertainty-weighted triangulation auxiliary loss logging.
        # The loss is already folded into ``epi_loss`` inside the model forward.
        if getattr(args, "use_v52_uncertainty_weighted_triangulation", False):
            v52_uwt_loss = getattr(model.module if hasattr(model, "module") else model, "_v52_uwt_loss", None)
            if v52_uwt_loss is not None and torch.isfinite(v52_uwt_loss):
                metrics["v52_uwt_loss"] = v52_uwt_loss.item()

        # Optional v53 physical-space calibration auxiliary loss logging.
        # The loss is already folded into ``epi_loss`` inside the model forward.
        if getattr(args, "use_v53_physical_space_calibration", False):
            v53_psc_loss = getattr(model.module if hasattr(model, "module") else model, "_v53_psc_loss", None)
            if v53_psc_loss is not None and torch.isfinite(v53_psc_loss):
                metrics["v53_psc_loss"] = v53_psc_loss.item()

        # Optional v54 physical-space calibration v2 auxiliary loss logging.
        if getattr(args, "use_v54_physical_space_calibration_v2", False):
            v54_psc2_loss = getattr(model.module if hasattr(model, "module") else model, "_v54_psc2_loss", None)
            if v54_psc2_loss is not None and torch.isfinite(v54_psc2_loss):
                metrics["v54_psc2_loss"] = v54_psc2_loss.item()

        # v48 adversarial / gradient-reversal domain loss (e.g. from DomainAdapterV48).
        if v48_domain_loss is not None:
            loss = loss + v48_domain_loss
            metrics["v48_domain_loss"] = v48_domain_loss.item()

        if args.visibility_loss_weight > 0.0:
            visible_target = (x[..., 2] > 0).float()
            visibility = torch.nan_to_num(visibility, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss
            metrics["vis_loss"] = vis_loss.item()

        if args.pa_loss_weight > 0.0:
            B, T, J, _ = pred_3d.shape
            pa_loss = procrustes_mse_loss(pred_3d.view(B * T, J, 3), y.view(B * T, J, 3))
            loss = loss + args.pa_loss_weight * pa_loss
            metrics["pa_loss"] = pa_loss.item()

        if args.uncertainty_loss_weight > 0.0:
            nll = uncertainty_nll_loss(pred_3d, x[..., :2], x[..., 2], K_aug, R_aug, t_aug, L)
            loss = loss + args.uncertainty_loss_weight * nll
            metrics["nll"] = nll.item()

        if args.temporal_loss_weight > 0.0 and pred_3d.shape[1] > 1:
            temp_loss, temp_metrics = temporal_consistency_loss(
                pred_3d,
                y,
                temporal_loss_weight=args.temporal_loss_weight,
                temporal_acceleration_weight=args.temporal_acceleration_weight,
            )
            loss = loss + temp_loss
            metrics.update(temp_metrics)

        # Optional v47 temporal smoothness loss on the refined trajectory.
        if (
            getattr(args, "use_v47_temporal_aggregation", False)
            and args.v47_temporal_loss_weight > 0.0
            and pred_3d.shape[1] > 1
        ):
            v47_temporal_loss = (pred_3d[:, 1:] - pred_3d[:, :-1]).abs().mean()
            loss = loss + args.v47_temporal_loss_weight * v47_temporal_loss
            metrics["v47_temporal_loss"] = v47_temporal_loss.item()

        # Optional v49-Lite temporal smoothness loss on the refined trajectory.
        if (
            getattr(args, "use_v49_lite_temporal_aggregation", False)
            and args.v49_lite_temporal_loss_weight > 0.0
            and pred_3d.shape[1] > 1
        ):
            v49_lite_temporal_loss = (pred_3d[:, 1:] - pred_3d[:, :-1]).abs().mean()
            loss = loss + args.v49_lite_temporal_loss_weight * v49_lite_temporal_loss
            metrics["v49_lite_temporal_loss"] = v49_lite_temporal_loss.item()

        # Optional v50 self-evolution feedback head auxiliary loss.
        # The head is now instantiated inside the model and its outputs are
        # exposed via attributes.
        if getattr(args, "use_v50_self_evolution_feedback_head", False):
            sefh_reliability = getattr(model, "last_sefh_reliability", None)
            sefh_log_var = getattr(model, "last_sefh_log_var", None)
            sefh_reproj = getattr(model, "last_sefh_reproj", None)
            sefh_temporal = getattr(model, "last_sefh_temporal", None)
            sefh_epipolar = getattr(model, "last_sefh_epipolar", None)
            if (
                sefh_reliability is not None
                and sefh_log_var is not None
                and sefh_reproj is not None
                and sefh_temporal is not None
                and sefh_epipolar is not None
            ):
                sefh_loss = compute_sefh_loss(
                    sefh_reliability,
                    sefh_log_var,
                    sefh_reproj,
                    sefh_temporal,
                    sefh_epipolar,
                    view_mask=view_mask,
                    loss_weight=args.v50_sefh_loss_weight,
                    residual_clip=args.v50_sefh_residual_clip,
                )
                if torch.isnan(sefh_loss).any() or torch.isinf(sefh_loss).any():
                    print(f"[trainer] v50 SEFH loss is NaN/Inf ({sefh_loss.item()}); skipping auxiliary loss", flush=True)
                else:
                    loss = loss + sefh_loss
                    metrics["v50_sefh_loss"] = sefh_loss.item()

                # Optional aleatoric uncertainty weighting of the per-joint error.
                aleatoric_weight = getattr(args, "v50_sefh_aleatoric_weight", 0.0)
                if aleatoric_weight > 0.0:
                    joint_err = (pred_3d - y).norm(dim=-1)  # (B, T, J)
                    aleatoric_loss = (torch.exp(-sefh_log_var) * joint_err + 0.5 * sefh_log_var).mean()
                    if not (torch.isnan(aleatoric_loss).any() or torch.isinf(aleatoric_loss).any()):
                        loss = loss + aleatoric_weight * aleatoric_loss
                        metrics["v50_sefh_aleatoric_loss"] = aleatoric_loss.item()

        # Optional v51 cross-domain sparse-view reliability (CDSVR) auxiliary loss.
        if getattr(args, "use_v51_cross_domain_sparse_view_reliability", False):
            cdsvr_reliability_refined = getattr(model, "last_cdsvr_reliability_refined", None)
            cdsvr_uncertainty_scale = getattr(model, "last_cdsvr_uncertainty_scale", None)
            if cdsvr_reliability_refined is not None and cdsvr_uncertainty_scale is not None:
                # Compute per-view reprojection residual from the current 3-D estimate.
                # project_points_3d_to_2d expects K/R/t of shape (B, V, ...).
                # The residual is detached because we only want to supervise the
                # refined reliability, not backprop through the projection math.
                with torch.no_grad():
                    uv_pred = project_points_3d_to_2d(
                        pred_3d, K_aug, R_aug, t_aug
                    )  # (B, T, V, J, 2)
                    reproj_err = (uv_pred - x[..., :2]).norm(dim=-1)  # (B, T, V, J)
                    # Average per-view residual across time and joints.
                    eps_view = reproj_err.mean(dim=(1, 3)).detach()  # (B, V)

                tau = getattr(args, "v51_cdsvr_uncertainty_temperature", 1.0)
                w_refined = torch.sigmoid(cdsvr_reliability_refined / tau)  # (B, V)

                # Weighted reprojection residual (Huber-like via sqrt(x^2 + delta^2)).
                delta = 0.1
                huber_res = torch.sqrt(eps_view ** 2 + delta ** 2)
                weighted_reproj = (w_refined * huber_res).mean()

                # Prevent collapsed uncertainty scales.
                entropy_scale = -torch.log(cdsvr_uncertainty_scale + 1e-8).mean()

                # Preserve view diversity (avoid all reliabilities collapsing).
                # Use a squared-deviation diversity term that is finite for any V>=1.
                mean_w = w_refined.mean(dim=-1, keepdim=True)
                view_div = ((w_refined - mean_w) ** 2).mean(dim=-1).mean()

                v51_cdsvr_loss_weight = getattr(args, "v51_cdsvr_loss_weight", 0.01)
                cdsvr_loss = v51_cdsvr_loss_weight * (
                    weighted_reproj - 0.5 * entropy_scale + 0.01 * view_div
                )
                if torch.isnan(cdsvr_loss).any() or torch.isinf(cdsvr_loss).any():
                    print(f"[trainer] v51 CDSVR loss is NaN/Inf ({cdsvr_loss.item()}); skipping auxiliary loss", flush=True)
                else:
                    loss = loss + cdsvr_loss
                    metrics["v51_cdsvr_loss"] = cdsvr_loss.item()

        if args.bone_loss_weight > 0.0 and parents is not None:
            bl = bone_length_loss(pred_3d, y, parents)
            loss = loss + args.bone_loss_weight * bl
            metrics["bone_loss"] = bl.item()

        if args.joint_limit_weight > 0.0 and parents is not None:
            jl = joint_limit_loss(
                pred_3d,
                parents,
                max_flexion_deg=args.joint_limit_max_flexion,
            )
            loss = loss + args.joint_limit_weight * jl
            metrics["joint_limit_loss"] = jl.item()

        if (
            args.temporal_bone_weight > 0.0
            and parents is not None
            and pred_3d.shape[1] > 1
        ):
            tbl = temporal_bone_length_loss(pred_3d, parents)
            loss = loss + args.temporal_bone_weight * tbl
            metrics["temporal_bone_loss"] = tbl.item()

        # Optional v40 skeleton-aware physical loss.
        if skeleton_physical_loss is not None:
            v40_phys = skeleton_physical_loss(
                pred_3d,
                target=y,
                epoch=getattr(model, "epoch", 1) - 1,
            )
            loss = loss + v40_phys
            metrics["v40_phys_loss"] = v40_phys.item()

        if args.attention_entropy_weight > 0.0:
            if entropy_loss_out is not None:
                loss = loss + args.attention_entropy_weight * entropy_loss_out
                metrics["entropy_loss"] = entropy_loss_out.item()
            else:
                ent = attention_entropy_loss(weights)
                loss = loss + args.attention_entropy_weight * ent
                metrics["entropy_loss"] = ent.item()

        if args.budget_loss_weight > 0.0:
            if budget_loss_out is not None:
                loss = loss + args.budget_loss_weight * budget_loss_out
                metrics["budget_loss"] = budget_loss_out.item()
            elif args.adaptive_view_k is not None and args.adaptive_view_k > 0:
                bgt = budget_loss(weights, args.adaptive_view_k)
                loss = loss + args.budget_loss_weight * bgt
                metrics["budget_loss"] = bgt.item()

        # Optional reprojection-loss warmup: linearly ramp all reprojection
        # weights from 0 over the first ``reproj_warmup_epochs`` epochs.
        if args.reproj_warmup_epochs > 0:
            current_epoch = getattr(model, "epoch", 1) - 1
            reproj_warmup_scale = min(1.0, max(0.0, current_epoch / args.reproj_warmup_epochs))
        else:
            reproj_warmup_scale = 1.0

        if args.reproj_loss_weight > 0.0 and reproj_warmup_scale > 0.0:
            reproj = _reprojection_loss(pred_3d, x[..., :2], K_aug, R_aug, t_aug, view_mask)
            loss = loss + (args.reproj_loss_weight * reproj_warmup_scale) * reproj
            metrics["reproj_loss"] = reproj.item()

        if args.aleatoric_reproj_loss_weight > 0.0 and reproj_warmup_scale > 0.0:
            ar_nll = uncertainty_nll_loss(pred_3d, x[..., :2], x[..., 2], K_aug, R_aug, t_aug, L)
            loss = loss + (args.aleatoric_reproj_loss_weight * reproj_warmup_scale) * ar_nll
            metrics["aleatoric_reproj_loss"] = ar_nll.item()

        # Optional monotonic multi-view loss: error with a subset of views should
        # not be better than using all views.  This is only meaningful when the
        # model is trained with variable-view subsets.
        if args.monotonic_loss_weight > 0.0 and view_mask is not None:
            with torch.no_grad():
                out_full = model(x, K=K_aug, R=R_aug, t=t_aug, view_mask=None)
            pred_full = out_full[0]
            err_subset = F.mse_loss(pred_3d, y)
            err_full = F.mse_loss(pred_full, y)
            active = view_mask.sum(dim=(1, 2)).max().item() / max(1, view_mask.shape[1])
            k_active = max(2, int(round(active)))
            margin = (args.monotonic_margin / 1000.0) / max(1, k_active)
            mono = torch.clamp(err_subset - err_full - margin, min=0.0)
            loss = loss + args.monotonic_loss_weight * mono
            metrics["monotonic_loss"] = mono.item()

        # Defensive: a single degenerate batch (singular geometry, corrupted data,
        # or numerical overflow) can otherwise explode the model and make the rest
        # of the epoch unusable.  Drop/clip such batches before backward.
        if not loss.isfinite():
            # Create a zero loss that still requires grad so backward succeeds.
            loss = torch.zeros((), device=device, requires_grad=True)
        loss = loss.clamp(max=1e6)

        return loss, metrics

    return compute_loss


def build_eval_metric():
    """Validation metric: MSE, MPJPE, and per-domain MPJPE."""

    def eval_metric(
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Dict[str, Any]:
        # Mixed loader returns an extra dataset_id field; keep it for per-domain metrics.
        if len(batch) == 6:
            x, y, K, R, t, dataset_id = batch
        else:
            x, y, K, R, t = batch
            dataset_id = None
        x = x.to(device)
        y = y.to(device)
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

        with torch.no_grad():
            forward_kwargs: Dict[str, Any] = {}
            if len(batch) == 6 and (
                getattr(model, "use_domain_embedding", False)
                or getattr(model, "use_v48_domain_generalization", False)
            ):
                forward_kwargs["domain_id"] = batch[5].to(device)
            out = model(x, K=K, R=R, t=t, **forward_kwargs)
            pred_3d = out[0]
            loss = F.mse_loss(pred_3d, y)
            mpjpe = (pred_3d - y).norm(dim=-1).mean()

            result: Dict[str, Any] = {"loss": loss, "mpjpe": mpjpe}

            # Per-domain MPJPE: helpful for spotting domain-gap regressions.
            if dataset_id is not None:
                domain_ids = batch[5].to(device).squeeze(-1).long()  # (B,)
                per_joint_error = (pred_3d - y).norm(dim=-1)  # (B, T, J)
                for d in torch.unique(domain_ids):
                    mask = domain_ids == d
                    if mask.any():
                        result[f"mpjpe_domain_{d.item()}"] = per_joint_error[mask].mean()

        return result

    return eval_metric


class OmniMultiViewTrainer(TrainerV2):
    """TrainerV2 pre-wired with the OmniMultiViewFusionV5 loss mix."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        args: Namespace,
        **kwargs: Any,
    ):
        self.args = args
        compute_loss = build_compute_loss(args)
        super().__init__(
            model,
            optimizer,
            device,
            compute_loss=compute_loss,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Multi-dataset helpers
# ---------------------------------------------------------------------------

def _inspect_npz(npz_path: str) -> Tuple[int, int]:
    """Return (n_views, n_joints) for a canonical .npz file."""
    data = np.load(npz_path)
    points_2d = data["points_2d"]
    return int(points_2d.shape[1]), int(points_2d.shape[2])


def _validate_dataset_consistency(
    train_files: List[str],
    val_files: List[str],
    graph_num_layers: int,
) -> Tuple[int, int]:
    """Ensure all files share the same (n_views, n_joints)."""
    all_files = list(train_files) + list(val_files)
    if not all_files:
        raise ValueError("No train or validation files found in the provided manifests.")

    shapes = []
    for f in all_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Manifest references missing file: {f}")
        shapes.append(_inspect_npz(f))

    unique = set(shapes)
    if len(unique) != 1:
        from collections import defaultdict
        grouped: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for f, s in zip(all_files, shapes):
            grouped[s].append(f)
        summary = "\n".join(
            f"  {s[1]} joints / {s[0]} views: {len(paths)} file(s)"
            for s, paths in sorted(grouped.items())
        )
        raise ValueError(
            "Mixed (n_views, n_joints) configurations detected across the selected manifests. "
            "OmniMultiViewFusionV5 is instantiated once per run and therefore requires all "
            "selected files to share the same view count and skeleton.\n"
            f"Detected configurations:\n{summary}\n"
            "Recommended fix: pass only manifests that share the same (n_views, n_joints), "
            "e.g. train on MPI-INF-3DHP only, or H36M only."
        )

    n_views, n_joints = shapes[0]

    if graph_num_layers > 0 and n_joints not in (17, 28):
        raise ValueError(
            f"OmniMultiViewFusionV5 graph attention only supports 17 or 28 joints, "
            f"but the selected data has {n_joints} joints. "
            f"Use --graph_num_layers 0 to disable graph attention for this skeleton."
        )

    return n_views, n_joints


def build_datasets(args: Namespace) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]:
    """Build train/val ConcatDatasets from manifest(s) and infer (n_views, n_joints)."""
    if args.smoke:
        K, R, t = _make_synthetic_cameras(n_views=4)
        n_joints = 17
        train_dataset: torch.utils.data.Dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=50, n_joints=n_joints, clip_len=args.clip_len
        )
        val_dataset: torch.utils.data.Dataset = SyntheticSmokeDataset(
            K, R, t, n_frames=30, n_joints=n_joints, clip_len=args.clip_len
        )
        return train_dataset, val_dataset, 4, n_joints

    if args.use_mixed_loader:
        if args.mixed_manifest is None:
            raise ValueError("--mixed_manifest is required when --use_mixed_loader is set")
        import yaml

        with open(args.mixed_manifest, "r") as f:
            mixed_cfg = yaml.safe_load(f)

        train_paths = mixed_cfg["train_paths"]
        train_names = mixed_cfg["train_names"]
        val_paths = mixed_cfg["val_paths"]
        val_names = mixed_cfg["val_names"]
        if not (len(train_paths) == len(train_names) and len(val_paths) == len(val_names)):
            raise ValueError("Mixed manifest: paths and names must have the same length")

        train_loader, val_loader = build_webbridge_mixed_dataloaders(
            train_paths=train_paths,
            train_names=train_names,
            val_paths=val_paths,
            val_names=val_names,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            train_samples=args.train_samples,
            val_stride=args.val_stride,
            num_workers=0,
        )
        # The mixed loader always pads to 14 views and maps to the 17-joint skeleton.
        return train_loader.dataset, val_loader.dataset, 14, 17

    if args.manifest:
        split = load_multi_dataset_manifest(args.manifest)
    else:
        split = {"train": args.train or [], "val": args.val or [], "test": []}

    train_files = split.get("train", [])
    val_files = split.get("val", [])

    if not train_files or not val_files:
        raise ValueError(
            "No train/val files found. Provide --manifest or both --train and --val."
        )

    n_views, n_joints = _validate_dataset_consistency(train_files, val_files, args.graph_num_layers)

    train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples) for tp in train_files]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(val_files[0], args.clip_len, stride=args.val_stride)

    if len(val_files) > 1:
        val_datasets = [TemporalClipDataset(vp, args.clip_len, stride=args.val_stride) for vp in val_files]
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    return train_dataset, val_dataset, n_views, n_joints


# ---------------------------------------------------------------------------
# Warm-start helpers
# ---------------------------------------------------------------------------

def load_warm_start(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint, tolerating extra/missing keys for warm-starting."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warm-start: missing keys (expected for new heads): {missing[:10]}")
    if unexpected:
        print(f"Warm-start: unexpected keys ignored: {unexpected[:10]}")
    print(f"Warm-started from {checkpoint_path}")


def freeze_old_params(model: torch.nn.Module) -> None:
    """Freeze everything except the new v5-specific heads."""
    new_prefixes = (
        "graph_joint_attention",
        "visibility_head",
        "omni_joint_attn",
        "multiscale_fusion",
        "camera_conditioning",
        "rotation_correction",
        "adaptive_view_selector",
        "kinematic_refiner",
        "skeleton_graph_residual",
        "camera_view_embedding",
        "set_view_aggregator",
    )
    for name, param in model.named_parameters():
        if not name.startswith(new_prefixes):
            param.requires_grad = False


def unfreeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_v47_base_params(model: torch.nn.Module) -> None:
    """Freeze everything except the v47 temporal aggregation head.

    Used to warm-start the temporal head while keeping the v46 (and older)
    per-frame backbone fixed so the head learns on stable per-frame poses.
    """
    for name, param in model.named_parameters():
        if "temporal_aggregation_v47" not in name:
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Train OmniMultiViewFusionV5 on one or more WebBridge datasets",
    )
    # Data
    parser.add_argument(
        "--manifest",
        type=str,
        action="append",
        default=None,
        help="Path to a YAML split manifest (can be passed multiple times). "
             "Default: configs/splits/webbridge_all_train.yaml",
    )
    parser.add_argument("--train", type=str, nargs="+", default=None, help="Train .npz files (legacy, overrides manifest train)")
    parser.add_argument("--val", type=str, default=None, help="Validation .npz file (legacy, overrides manifest val)")
    parser.add_argument("--smoke", action="store_true", help="1-epoch CPU/GPU smoke test on synthetic data")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes (default 0)")
    parser.add_argument("--use_mixed_loader", action="store_true", help="Use the WebBridge mixed-dataset loader (H36M+MPI, 17 joints/14 views)")
    parser.add_argument("--mixed_manifest", type=str, default=None, help="YAML manifest for mixed loader (train_paths/train_names/val_paths/val_names)")
    # Model
    parser.add_argument("--d", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--residual_hidden", type=int, default=128, help="Residual MLP hidden size")
    parser.add_argument("--n_st_layers", type=int, default=2, help="Spatio-temporal transformer layers")
    parser.add_argument("--graph_num_layers", type=int, default=1, help="Graph-joint attention layers")
    parser.add_argument("--n_joint_layers", type=int, default=0, help="Dense joint-level transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--epipolar_loss_weight", type=float, default=0.05, help="Epipolar loss weight passed to the model")
    # v4 toggles
    parser.add_argument("--use_multiscale_fusion", type=lambda x: x.lower() == "true", default=True, help="Enable hierarchical multi-scale fusion")
    parser.add_argument("--use_adaptive_multiscale_fusion", action="store_true", help="Enable adaptive scale-selective multi-scale fusion")
    parser.add_argument("--use_camera_conditioning", type=lambda x: x.lower() == "true", default=True, help="Enable camera conditioning")
    parser.add_argument("--use_epipolar_bias", type=lambda x: x.lower() == "true", default=True, help="Enable epipolar-biased ST transformer")
    parser.add_argument("--use_context_visibility", type=lambda x: x.lower() == "true", default=True, help="Use context-aware visibility head")
    parser.add_argument("--use_skeleton_residual", type=lambda x: x.lower() == "true", default=True, help="Use skeleton-graph residual refiner")
    parser.add_argument("--use_skeleton_residual_v31", type=lambda x: x.lower() == "true", default=False, help="Use v31 per-joint gated skeleton-graph residual refiner")
    parser.add_argument("--use_kinematic_refiner", type=lambda x: x.lower() == "true", default=False, help="Use kinematic-chain final refiner")
    parser.add_argument("--use_adaptive_view_selection", type=lambda x: x.lower() == "true", default=False, help="Use adaptive view selection")
    parser.add_argument("--use_rotation_correction", type=lambda x: x.lower() == "true", default=False, help="Use rotation correction head")
    parser.add_argument("--use_entropy_regularization", type=lambda x: x.lower() == "true", default=False, help="Enable entropy regularisation inside the model")
    parser.add_argument("--adaptive_view_k", type=int, default=None, help="Target k for adaptive view selection")
    # v5 toggles
    parser.add_argument("--use_camera_view_embedding", action="store_true", help="Use camera-conditioned view embedding instead of learned view_pos_embed")
    parser.add_argument("--use_camera_view_embedding_v31", action="store_true", help="Use v31 geometry-aware pairwise camera view embedding")
    parser.add_argument("--use_set_view_aggregator", action="store_true", help="Use set-transformer (ISAB) view aggregator before the time+view transformer")
    parser.add_argument("--camera_view_embedding_hidden", type=int, default=32, help="Hidden dimension of camera-conditioned view embedding MLP")
    parser.add_argument("--set_view_n_isab_layers", type=int, default=2, help="Number of ISAB layers in set aggregator")
    parser.add_argument("--set_view_num_inducing_points", type=int, default=32, help="Number of inducing points per ISAB")
    parser.add_argument("--set_view_dropout", type=float, default=0.0, help="Dropout in set aggregator")
    # v6 toggles
    parser.add_argument("--use_perceiver_aggregator", action="store_true", help="Use Perceiver-style view aggregator instead of ISAB")
    parser.add_argument("--perceiver_n_latents", type=int, default=16, help="Number of latent vectors in Perceiver aggregator")
    parser.add_argument("--perceiver_n_layers", type=int, default=2, help="Number of Perceiver latent layers")
    parser.add_argument("--perceiver_n_heads", type=int, default=4, help="Number of attention heads in Perceiver aggregator")
    parser.add_argument("--perceiver_dropout", type=float, default=0.0, help="Dropout in Perceiver aggregator")
    parser.add_argument("--use_cross_view_transformer_v17", action="store_true", help="Use v17 geometric cross-view transformer for view aggregation")
    parser.add_argument("--use_deformable_cross_view_attention_v18", action="store_true", help="Use v18 sparse epipolar-aware cross-view attention")
    parser.add_argument("--deformable_attention_use_topk_st", action="store_true", help="Use straight-through top-k sampling in v18 deformable attention (default: soft attention)")
    parser.add_argument("--use_temporal_perceiver_v19", action="store_true", help="Use v19 temporal Perceiver refiner on final 3D poses sequence")
    parser.add_argument("--use_diffusion_refiner_v20", action="store_true", help="Use v20 diffusion-based residual refiner instead of deterministic MLP")
    parser.add_argument("--num_diffusion_steps", type=int, default=10, help="Number of diffusion timesteps for v20 refiner")
    parser.add_argument("--use_neural_bundle_adjustment_v21", action="store_true", help="Use v21 neural bundle-adjustment pose/camera refiner")
    parser.add_argument("--use_kinematic_anthropometric_prior_v22", action="store_true", help="Use v22 kinematic anthropometric prior (SMPL-free bone-length prior)")
    parser.add_argument("--kap_loss_weight", type=float, default=0.01, help="Weight for v22 KAP loss")
    # v25 toggles
    parser.add_argument("--use_multiview_geometry_fusion_v25", action="store_true", help="Use v25 multi-view geometry fusion (ray-token + epipolar cross-view attention + learned depth triangulation)")
    parser.add_argument("--v25_geom_loss_weight", type=float, default=0.1, help="Weight for v25 geometry consistency loss")
    parser.add_argument("--v25_dropout", type=float, default=0.1, help="Dropout rate for v25 ray tokenizer and depth-proposal MLPs")
    parser.add_argument("--v25_use_geometry_attention", action="store_true", default=True, help="Use geometry-aware cross-view attention in v25")
    parser.add_argument("--no_v25_use_geometry_attention", dest="v25_use_geometry_attention", action="store_false", help="Disable geometry-aware cross-view attention in v25")
    parser.add_argument("--v25_use_learned_depth_triangulation", action="store_true", default=True, help="Use learned depth triangulation in v25")
    parser.add_argument("--no_v25_use_learned_depth_triangulation", dest="v25_use_learned_depth_triangulation", action="store_false", help="Disable learned depth triangulation in v25")
    parser.add_argument("--v25_use_geometry_bundle_adjustment", action="store_true", default=True, help="Use geometry bundle adjustment in v25")
    parser.add_argument("--no_v25_use_geometry_bundle_adjustment", dest="v25_use_geometry_bundle_adjustment", action="store_false", help="Disable geometry bundle adjustment in v25")
    parser.add_argument("--v25_use_camera_joint_graph", action="store_true", default=False, help="Use camera-joint graph in v25")
    parser.add_argument("--v25_use_outlier_view_detector", action="store_true", default=False, help="Use robust outlier-view detection and down-weighting in v25")
    parser.add_argument("--v25_outlier_z_thresh", type=float, default=3.0, help="Robust z-score threshold for v25 outlier-view detector")
    parser.add_argument("--v25_outlier_soft_beta", type=float, default=1.0, help="Softness of exponential down-weighting for v25 outlier-view detector")
    parser.add_argument("--use_temporal_geometry_fusion_v26", action="store_true", default=False, help="Use v26 temporal geometry fusion instead of v25")
    parser.add_argument("--v26_temporal_window", type=int, default=3, help="Temporal window size for v26 (must be odd)")
    parser.add_argument("--v26_temporal_attention_residual_gate_init", type=float, default=0.0, help="Initial value of the residual gate in v26 temporal attention (0.0 = identity at init)")
    parser.add_argument("--use_uncertainty_depth_proposals_v27", action="store_true", default=False, help="Use v27 uncertainty-aware depth-proposal triangulation head in v25/v26")
    parser.add_argument("--v27_uncertainty_loss_weight", type=float, default=0.01, help="Weight for v27 uncertainty regularisation loss")
    parser.add_argument("--v27_udp_n_mixtures", type=int, default=1, help="Number of Gaussian mixture components for v27 depth proposals (default 1=single Gaussian)")
    # v45 adaptive geometry fusion for v25 triangulation
    parser.add_argument("--use_v45_adaptive_geometry_fusion", action="store_true", default=False, help="Use v45 adaptive geometry fusion to learn triangulation weights in v25")
    parser.add_argument("--v45_adaptive_weight_type", type=str, default="per_view", choices=["per_view", "per_joint", "per_view_joint"], help="Granularity of v45 adaptive triangulation weights")
    parser.add_argument("--v45_adaptive_weight_hidden", type=int, default=32, help="Hidden dimension of the v45 adaptive weight MLP")
    parser.add_argument("--v45_adaptive_weight_n_layers", type=int, default=1, help="Number of layers in the v45 adaptive weight MLP")
    # v46 sparse-view generalization
    parser.add_argument("--use_v46_sparse_view_generalization", action="store_true", default=False, help="Use v46 sparse-view generalization head")
    parser.add_argument("--v46_svg_view_dropout_prob", type=float, default=0.3, help="Probability of dropping each view during v46 training")
    parser.add_argument("--v46_svg_min_views", type=int, default=2, help="Minimum number of retained views during v46 dropout")
    parser.add_argument("--v46_svg_hidden", type=int, default=64, help="Hidden dimension of the v46 reliability MLP")
    parser.add_argument("--v46_svg_use_curriculum", action="store_true", default=True, help="Use curriculum for v46 view dropout (default True)")
    parser.add_argument("--no_v46_svg_use_curriculum", dest="v46_svg_use_curriculum", action="store_false", help="Disable v46 view-dropout curriculum")
    # v47 temporal aggregation
    parser.add_argument("--use_v47_temporal_aggregation", action="store_true", default=False, help="Use v47 temporal aggregation head on top of v46 sparse-view generalization")
    parser.add_argument("--v47_temporal_d_model", type=int, default=64, help="Hidden dimension of the v47 temporal aggregation transformer")
    parser.add_argument("--v47_temporal_n_heads", type=int, default=4, help="Number of attention heads in v47 temporal aggregation")
    parser.add_argument("--v47_temporal_num_layers", type=int, default=2, help="Number of transformer layers in v47 temporal aggregation")
    parser.add_argument("--v47_temporal_window", type=int, default=None, help="Temporal window size for v47 (None = full-clip attention)")
    parser.add_argument("--v47_temporal_dropout", type=float, default=0.1, help="Dropout probability in v47 temporal aggregation")
    parser.add_argument("--v47_temporal_loss_weight", type=float, default=0.01, help="Weight for v47 temporal smoothness loss")
    parser.add_argument("--v47_use_view_count_conditioning", action="store_true", default=True, help="Condition v47 temporal aggregation on the number of active views per frame")
    parser.add_argument("--no_v47_use_view_count_conditioning", dest="v47_use_view_count_conditioning", action="store_false", help="Disable view-count conditioning in v47 temporal aggregation")
    parser.add_argument("--v47_freeze_base_epochs", type=int, default=0, help="Freeze base model parameters for the first N epochs when v47 is enabled (0 disables)")
    # v49-Lite lightweight causal temporal aggregation
    parser.add_argument("--use_v49_lite_temporal_aggregation", action="store_true", default=False, help="Use v49-Lite lightweight causal Conv1D temporal aggregation instead of v47")
    parser.add_argument("--v49_lite_temporal_d_model", type=int, default=32, help="Hidden dimension of the v49-Lite temporal aggregation")
    parser.add_argument("--v49_lite_temporal_num_layers", type=int, default=2, help="Number of causal Conv1D blocks in v49-Lite temporal aggregation")
    parser.add_argument("--v49_lite_temporal_kernel_size", type=int, default=3, help="Kernel size for v49-Lite causal Conv1D (odd)")
    parser.add_argument("--v49_lite_temporal_dropout", type=float, default=0.1, help="Dropout probability in v49-Lite temporal aggregation")
    parser.add_argument("--v49_lite_temporal_residual_gate_init", type=float, default=0.0, help="Initial residual gate for v49-Lite temporal aggregation (0 = identity at init)")
    parser.add_argument("--v49_lite_temporal_use_view_count_conditioning", action="store_true", default=True, help="Condition v49-Lite temporal aggregation on the number of active views per frame")
    parser.add_argument("--no_v49_lite_temporal_use_view_count_conditioning", dest="v49_lite_temporal_use_view_count_conditioning", action="store_false", help="Disable view-count conditioning in v49-Lite temporal aggregation")
    parser.add_argument("--v49_lite_temporal_loss_weight", type=float, default=0.01, help="Weight for v49-Lite temporal smoothness loss")
    # v50 self-evolution feedback head
    parser.add_argument("--use_v50_self_evolution_feedback_head", action="store_true", default=False, help="Use v50 self-evolution feedback head auxiliary loss")
    parser.add_argument("--v50_sefh_hidden", type=int, default=64, help="Hidden dimension of the v50 feedback-head MLP")
    parser.add_argument("--v50_sefh_num_layers", type=int, default=2, help="Number of layers in the v50 feedback-head MLP")
    parser.add_argument("--v50_sefh_dropout", type=float, default=0.1, help="Dropout probability in the v50 feedback-head MLP")
    parser.add_argument("--v50_sefh_reproj_weight", type=float, default=1.0, help="Weight for the reprojection residual branch in v50")
    parser.add_argument("--v50_sefh_temporal_weight", type=float, default=0.5, help="Weight for the temporal residual branch in v50")
    parser.add_argument("--v50_sefh_epipolar_weight", type=float, default=0.5, help="Weight for the epipolar residual branch in v50")
    parser.add_argument("--v50_sefh_loss_weight", type=float, default=0.01, help="Weight for the v50 auxiliary loss")
    parser.add_argument("--v50_sefh_residual_clip", type=float, default=50.0, help="Clip residuals before feeding into the v50 loss")
    parser.add_argument("--v50_sefh_identity_init_gate", action="store_true", default=True, help="Initialize the v50 reliability gate near identity")
    parser.add_argument("--no_v50_sefh_identity_init_gate", dest="v50_sefh_identity_init_gate", action="store_false", help="Disable v50 identity init gate")
    parser.add_argument("--v50_sefh_aleatoric_weight", type=float, default=0.0, help="Weight for aleatoric uncertainty loss on the per-joint error (0 disables)")
    # v51 cross-domain sparse-view reliability (CDSVR)
    parser.add_argument("--use_v51_cross_domain_sparse_view_reliability", action="store_true", default=False, help="Use v51 cross-domain sparse-view reliability head on top of v50 SEFH")
    parser.add_argument("--v51_cdsvr_hidden", type=int, default=64, help="Hidden dimension of the v51 CDSVR cross-attention block")
    parser.add_argument("--v51_cdsvr_num_heads", type=int, default=4, help="Number of attention heads in v51 CDSVR")
    parser.add_argument("--v51_cdsvr_dropout", type=float, default=0.1, help="Dropout probability in v51 CDSVR")
    parser.add_argument("--v51_cdsvr_offset_min", type=float, default=0.05, help="Minimum floor for refined reliability weight in v51 CDSVR")
    parser.add_argument("--v51_cdsvr_use_domain_label", action="store_true", default=True, help="Use one-hot domain label in v51 CDSVR")
    parser.add_argument("--no_v51_cdsvr_use_domain_label", dest="v51_cdsvr_use_domain_label", action="store_false", help="Disable domain label in v51 CDSVR")
    parser.add_argument("--v51_cdsvr_uncertainty_temperature", type=float, default=1.0, help="Temperature for refined reliability sigmoid in v51 CDSVR")
    parser.add_argument("--v51_cdsvr_identity_init_gate", action="store_true", default=True, help="Initialize v51 CDSVR near identity")
    parser.add_argument("--no_v51_cdsvr_identity_init_gate", dest="v51_cdsvr_identity_init_gate", action="store_false", help="Disable v51 CDSVR identity init gate")
    parser.add_argument("--v51_cdsvr_loss_weight", type=float, default=0.01, help="Weight for the v51 CDSVR auxiliary loss")
    # v51 domain-agnostic ensemble of pose experts
    parser.add_argument("--use_v51_domain_agnostic_ensemble", action="store_true", default=False, help="Use v51 domain-agnostic ensemble of pose experts")
    parser.add_argument("--v51_dae_hidden", type=int, default=64, help="Hidden dimension of the v51 ensemble evidence MLP")
    parser.add_argument("--v51_dae_num_layers", type=int, default=2, help="Number of layers in the v51 ensemble evidence MLP")
    parser.add_argument("--v51_dae_dropout", type=float, default=0.1, help="Dropout probability in the v51 ensemble evidence MLP")
    parser.add_argument("--v51_dae_n_experts", type=int, default=2, help="Number of experts in the v51 ensemble (default 2: geometry + residual)")
    parser.add_argument("--v51_dae_identity_bypass", action="store_true", default=True, help="Add uniform bypass to the v51 ensemble weights at init")
    parser.add_argument("--no_v51_dae_identity_bypass", dest="v51_dae_identity_bypass", action="store_false", help="Disable v51 ensemble identity bypass")
    parser.add_argument("--v51_dae_min_weight", type=float, default=0.05, help="Minimum per-expert weight in the v51 ensemble")
    parser.add_argument("--v51_dae_loss_weight", type=float, default=0.005, help="Weight for the v51 ensemble diversity auxiliary loss")
    # v52 uncertainty-weighted triangulation
    parser.add_argument("--use_v52_uncertainty_weighted_triangulation", action="store_true", default=False, help="Use v52 uncertainty-weighted triangulation")
    parser.add_argument("--v52_uwt_hidden", type=int, default=64, help="Hidden dimension of the v52 precision MLP")
    parser.add_argument("--v52_uwt_n_layers", type=int, default=2, help="Number of layers in the v52 precision MLP")
    parser.add_argument("--v52_uwt_weight_type", type=str, default="per_view_joint", choices=["per_view_joint", "per_view", "per_joint"], help="v52 precision weight type")
    parser.add_argument("--v52_uwt_temperature", type=float, default=1.0, help="Temperature for v52 precision sigmoid")
    parser.add_argument("--v52_uwt_use_geometry_bias", action="store_true", default=True, help="Use geometry bias in v52 precision MLP")
    parser.add_argument("--no_v52_uwt_use_geometry_bias", dest="v52_uwt_use_geometry_bias", action="store_false", help="Disable geometry bias in v52 precision MLP")
    parser.add_argument("--v52_uwt_use_feature_bias", action="store_true", default=True, help="Use feature bias in v52 precision MLP")
    parser.add_argument("--no_v52_uwt_use_feature_bias", dest="v52_uwt_use_feature_bias", action="store_false", help="Disable feature bias in v52 precision MLP")
    parser.add_argument("--v52_uwt_identity_init", action="store_true", default=True, help="Zero-initialise v52 precision MLP final layer")
    parser.add_argument("--no_v52_uwt_identity_init", dest="v52_uwt_identity_init", action="store_false", help="Disable v52 identity initialisation")
    parser.add_argument("--v52_uwt_min_weight", type=float, default=0.05, help="Minimum v52 triangulation weight")
    parser.add_argument("--v52_uwt_loss_weight", type=float, default=0.01, help="Weight for the v52 UWT auxiliary loss")
    parser.add_argument("--v52_uwt_damping", type=float, default=1e-4, help="Ridge damping for v52 weighted DLT")
    parser.add_argument("--v52_uwt_warmup_epochs", type=int, default=0, help="Epochs before v52 UWT loss is active")
    # v53 physical-space calibration
    parser.add_argument("--use_v53_physical_space_calibration", action="store_true", default=False, help="Use v53 physical-space calibration")
    parser.add_argument("--v53_psc_hidden", type=int, default=64, help="Hidden dimension of the v53 residual MLP")
    parser.add_argument("--v53_psc_n_layers", type=int, default=2, help="Number of layers in the v53 residual MLP")
    parser.add_argument("--v53_psc_identity_init", action="store_true", default=True, help="Zero-initialise the v53 residual MLP final layer")
    parser.add_argument("--no_v53_psc_identity_init", dest="v53_psc_identity_init", action="store_false", help="Disable v53 identity initialisation")
    parser.add_argument("--v53_psc_residual_gate_init", type=float, default=-6.0, help="Initial logit for the v53 residual gate")
    parser.add_argument("--v53_psc_use_uwt_weights", action="store_true", default=True, help="Use v52 UWT weights in the v53 floor head")
    parser.add_argument("--no_v53_psc_use_uwt_weights", dest="v53_psc_use_uwt_weights", action="store_false", help="Disable v52 UWT weights in the v53 floor head")
    parser.add_argument("--v53_psc_use_floor", action="store_true", default=True, help="Enable the v53 floor-plane head")
    parser.add_argument("--no_v53_psc_use_floor", dest="v53_psc_use_floor", action="store_false", help="Disable the v53 floor-plane head")
    parser.add_argument("--v53_psc_use_bone_scale", action="store_true", default=True, help="Enable the v53 canonical bone-length head")
    parser.add_argument("--no_v53_psc_use_bone_scale", dest="v53_psc_use_bone_scale", action="store_false", help="Disable the v53 canonical bone-length head")
    parser.add_argument("--v53_psc_loss_weight", type=float, default=1.0, help="Weight for the v53 PSC auxiliary loss")
    parser.add_argument("--v53_psc_floor_weight", type=float, default=0.01, help="Weight for the v53 floor loss term")
    parser.add_argument("--v53_psc_bone_weight", type=float, default=0.1, help="Weight for the v53 bone-length loss term")
    parser.add_argument("--v53_psc_reproj_weight", type=float, default=0.1, help="Weight for the v53 reprojection consistency term")
    parser.add_argument("--v53_psc_warmup_epochs", type=int, default=0, help="Epochs before the v53 PSC loss is active")
    parser.add_argument("--v53_psc_min_visible_views", type=int, default=2, help="Minimum visible views for a joint to contribute to v53 losses")
    # v54 physical-space calibration v2
    parser.add_argument("--use_v54_physical_space_calibration_v2", action="store_true", default=False, help="Use v54 physical-space calibration v2")
    parser.add_argument("--v54_psc2_hidden", type=int, default=64, help="Hidden dimension of the v54 PSC-v2 refiner")
    parser.add_argument("--v54_psc2_n_layers", type=int, default=2, help="Number of layers in the v54 PSC-v2 refiner")
    parser.add_argument("--v54_psc2_num_domains", type=int, default=8, help="Number of domains for v54 canonical bone scales")
    parser.add_argument("--v54_psc2_use_floor", action="store_true", default=True, help="Enable the v54 floor-plane head")
    parser.add_argument("--no_v54_psc2_use_floor", dest="v54_psc2_use_floor", action="store_false", help="Disable the v54 floor-plane head")
    parser.add_argument("--v54_psc2_use_contact", action="store_true", default=True, help="Enable the v54 velocity-gated contact loss")
    parser.add_argument("--no_v54_psc2_use_contact", dest="v54_psc2_use_contact", action="store_false", help="Disable the v54 contact loss")
    parser.add_argument("--v54_psc2_use_bone_scale", action="store_true", default=True, help="Enable the v54 canonical bone-length head")
    parser.add_argument("--no_v54_psc2_use_bone_scale", dest="v54_psc2_use_bone_scale", action="store_false", help="Disable the v54 bone-length head")
    parser.add_argument("--v54_psc2_use_temporal_smoothness", action="store_true", default=True, help="Enable the v54 temporal smoothness loss")
    parser.add_argument("--no_v54_psc2_use_temporal_smoothness", dest="v54_psc2_use_temporal_smoothness", action="store_false", help="Disable the v54 temporal smoothness loss")
    parser.add_argument("--v54_psc2_use_gnn", action="store_true", default=True, help="Use the v54 skeleton-graph refiner (fallback: per-joint MLP)")
    parser.add_argument("--no_v54_psc2_use_gnn", dest="v54_psc2_use_gnn", action="store_false", help="Disable the v54 skeleton-graph refiner")
    parser.add_argument("--v54_psc2_gnn_layers", type=int, default=1, help="Number of v54 GNN layers")
    parser.add_argument("--v54_psc2_identity_init", action="store_true", default=True, help="Zero-initialise the v54 residual layers and gate")
    parser.add_argument("--no_v54_psc2_identity_init", dest="v54_psc2_identity_init", action="store_false", help="Disable v54 identity initialisation")
    parser.add_argument("--v54_psc2_residual_gate_init", type=float, default=-6.0, help="Initial logit for the v54 residual gate")
    parser.add_argument("--v54_psc2_loss_weight", type=float, default=1.0, help="Weight for the v54 PSC-v2 auxiliary loss")
    parser.add_argument("--v54_psc2_floor_weight", type=float, default=0.01, help="Weight for the v54 floor loss term")
    parser.add_argument("--v54_psc2_bone_weight", type=float, default=0.05, help="Weight for the v54 bone-length loss term")
    parser.add_argument("--v54_psc2_contact_weight", type=float, default=0.01, help="Weight for the v54 contact loss term")
    parser.add_argument("--v54_psc2_temporal_weight", type=float, default=0.01, help="Weight for the v54 temporal smoothness loss term")
    parser.add_argument("--v54_psc2_reproj_weight", type=float, default=0.1, help="Weight for the v54 reprojection consistency term")
    parser.add_argument("--v54_psc2_contact_velocity_thresh", type=float, default=0.3, help="Foot velocity threshold (m/s) for the v54 contact loss")
    parser.add_argument("--v54_psc2_min_visible_views", type=int, default=2, help="Minimum visible views for a joint to contribute to v54 losses")
    parser.add_argument("--v54_psc2_warmup_epochs", type=int, default=0, help="Epochs before the v54 PSC-v2 loss is active")
    parser.add_argument("--use_v51_test_time_self_evolution_refiner", action="store_true", default=False, help="Use v51 test-time self-evolution refiner at inference")
    parser.add_argument("--v51_tta_num_steps", type=int, default=3, help="v51 TTSER gradient steps per clip")
    parser.add_argument("--v51_tta_lr", type=float, default=1e-3, help="v51 TTSER Adam learning rate")
    parser.add_argument("--v51_tta_hidden", type=int, default=32, help="v51 TTSER update MLP hidden dimension")
    parser.add_argument("--v51_tta_reproj_weight", type=float, default=1.0, help="v51 TTSER reprojection-loss weight")
    parser.add_argument("--v51_tta_temporal_weight", type=float, default=0.5, help="v51 TTSER temporal-loss weight")
    parser.add_argument("--v51_tta_bone_weight", type=float, default=0.1, help="v51 TTSER bone-loss weight")
    parser.add_argument("--v51_tta_entropy_weight", type=float, default=0.01, help="v51 TTSER entropy regularisation weight")
    parser.add_argument("--v51_tta_min_view_rel", type=float, default=0.05, help="v51 TTSER minimum per-view reliability")
    parser.add_argument("--v51_tta_max_view_rel", type=float, default=1.0, help="v51 TTSER maximum per-view reliability")
    parser.add_argument("--no_v51_tta_use_sefh_init", dest="v51_tta_use_sefh_init", action="store_false", help="Disable v51 TTSER v50 SEFH initialisation")
    # v48 domain generalization
    parser.add_argument("--use_v48_domain_generalization", action="store_true", default=False, help="Use v48 domain generalization (DDWL + optional domain FiLM / adversarial loss)")
    parser.add_argument("--v48_dg_hidden", type=int, default=64, help="Hidden dimension of the v48 domain adapter")
    parser.add_argument("--v48_dg_grl_lambda", type=float, default=0.1, help="Gradient-reversal lambda for the v48 domain discriminator")
    parser.add_argument("--v48_dg_use_domain_film", action="store_true", default=True, help="Use domain-conditional FiLM offsets in v48")
    parser.add_argument("--no_v48_dg_use_domain_film", dest="v48_dg_use_domain_film", action="store_false", help="Disable domain-conditional FiLM offsets in v48")
    parser.add_argument("--v48_dg_use_ddwl", action="store_true", default=True, help="Use v48 dynamic domain-difficulty-weighted loss (DDWL)")
    parser.add_argument("--no_v48_dg_use_ddwl", dest="v48_dg_use_ddwl", action="store_false", help="Disable v48 DDWL")
    parser.add_argument("--v48_dg_ddwl_temperature", type=float, default=2.0, help="Temperature for v48 DDWL adaptive weights")
    parser.add_argument("--v48_dg_ddwl_warmup_epochs", type=int, default=1, help="Epochs of uniform DDWL weights before adapting")
    parser.add_argument("--v48_3dpw_actual_val_paths", type=str, default=None, help="Comma-separated list of 3DPW actual-mode .npz files for validation")
    parser.add_argument("--v48_dropout_per_domain", type=str, default=None, help="JSON dict of per-domain view-dropout probabilities, e.g. '{\"0\": 0.30, \"1\": 0.30, \"5\": 0.15}'")
    parser.add_argument("--use_test_time_self_evolution_v27", action="store_true", default=False, help="Use v27 test-time self-evolution at inference")
    parser.add_argument("--use_physical_space_alignment_v28", action="store_true", default=False, help="Use v28 physical-space alignment refiner")
    parser.add_argument("--use_physical_space_alignment_v32", action="store_true", default=False, help="Use v32 root-centered per-joint bounded physical-space alignment")
    parser.add_argument("--v28_floor_loss_weight", type=float, default=0.0, help="Weight for v28 floor consistency loss")
    parser.add_argument("--v28_bone_temporal_weight", type=float, default=0.0, help="Weight for v28 bone-length temporal consistency loss")
    parser.add_argument("--v27_tte_n_iters", type=int, default=3, help="Number of iterations for v27 test-time self-evolution")
    parser.add_argument("--v27_tte_sigma_reproj", type=float, default=5.0, help="Cauchy kernel scale (pixels) for v27 self-evolution")
    parser.add_argument("--v27_tte_residual_thresh_mm", type=float, default=0.5, help="Early-stop threshold (mm) for v27 self-evolution")
    # v29 arguments
    parser.add_argument("--use_hierarchical_multiview_v29", action="store_true", default=False, help="Use v29 hierarchical multi-scale view encoder")
    parser.add_argument("--v29_n_heads", type=int, default=4, help="Number of attention heads for v29 hierarchical encoder")
    parser.add_argument("--v29_n_part_layers", type=int, default=1, help="Number of part-scale attention layers for v29")
    parser.add_argument("--use_hierarchical_multiview_v30", action="store_true", default=False, help="Use v30 hardened hierarchical multi-scale view encoder")
    parser.add_argument("--v30_n_heads", type=int, default=4, help="Number of attention heads for v30 hierarchical encoder")
    parser.add_argument("--v30_n_part_layers", type=int, default=1, help="Number of part-scale attention layers for v30")
    parser.add_argument("--use_hierarchical_multiview_v31", action="store_true", default=False, help="Use v31 geometry-biased hierarchical multi-scale view encoder")
    parser.add_argument("--v31_geometry_bias", action="store_true", default=True, help="Enable geometry bias in v31 encoder")
    parser.add_argument("--v31_use_ray_attention", action="store_true", default=False, help="Add ray-conditioned camera embedding to v31 encoder tokens")
    # v32 temporal trajectory consistency
    parser.add_argument("--use_trajectory_consistency_v32", action="store_true", default=False, help="Use v32 1-D temporal CNN trajectory-consistency refiner")
    parser.add_argument("--v32_smooth_weight", type=float, default=1e-3, help="Weight for v32 trajectory smoothness loss")
    parser.add_argument("--v32_drift_weight", type=float, default=1e-2, help="Weight for v32 drift guard loss")
    # v33 uncertainty-aware triangulation
    parser.add_argument("--use_uncertainty_aware_triangulation_v33", action="store_true", help="Use v33 uncertainty-aware triangulation head")
    parser.add_argument("--v33_uat_loss_weight", type=float, default=0.01, help="Weight for v33 uncertainty NLL loss")
    parser.add_argument("--v33_uat_log_var_min", type=float, default=-10.0, help="Min log-variance clamp")
    parser.add_argument("--v33_uat_log_var_max", type=float, default=10.0, help="Max log-variance clamp")
    parser.add_argument("--v33_uat_covariance_hidden", type=int, default=64, help="Hidden dim of the uncertainty-prediction MLP")
    # v33 outlier-view rejection
    parser.add_argument("--use_outlier_view_rejection_v33", action="store_true", default=False, help="Use v33 learned outlier-view detector in the main DLT path")
    parser.add_argument("--v33_outlier_z_thresh", type=float, default=3.0, help="Base z-score threshold for v33 outlier detector")
    parser.add_argument("--v33_outlier_soft_beta", type=float, default=1.0, help="Softness of exponential down-weighting for v33 outlier detector")
    parser.add_argument("--v33_outlier_use_feature_gate", action="store_true", default=True, help="Use feature-aware residual adjustment in v33 outlier detector")
    parser.add_argument("--no_v33_outlier_use_feature_gate", dest="v33_outlier_use_feature_gate", action="store_false", help="Disable feature-aware residual adjustment in v33 outlier detector")
    parser.add_argument("--v33_outlier_use_part_scale", action="store_true", default=True, help="Use per-body-part scales in v33 outlier detector")
    parser.add_argument("--no_v33_outlier_use_part_scale", dest="v33_outlier_use_part_scale", action="store_false", help="Disable per-body-part scales in v33 outlier detector")
    parser.add_argument("--v33_outlier_use_domain_scale", action="store_true", default=True, help="Use per-domain scales in v33 outlier detector")
    parser.add_argument("--no_v33_outlier_use_domain_scale", dest="v33_outlier_use_domain_scale", action="store_false", help="Disable per-domain scales in v33 outlier detector")
    parser.add_argument("--v33_outlier_supervised_weight", type=float, default=0.1, help="Weight for the optional supervised BCE loss in v33 outlier detector")
    parser.add_argument("--v33_outlier_feature_hidden", type=int, default=64, help="Hidden dim of the feature-aware MLP in v33 outlier detector")
    # v33 ray-conditioned cross-view attention
    parser.add_argument("--use_ray_conditioned_attention_v33", action="store_true", default=False, help="Use v33 ray-conditioned cross-view attention after the v31 hierarchical encoder")
    parser.add_argument("--v33_n_heads", type=int, default=4, help="Number of attention heads for v33 ray-conditioned attention")
    parser.add_argument("--v33_n_layers", type=int, default=2, help="Number of layers for v33 ray-conditioned attention")
    parser.add_argument("--v33_dropout", type=float, default=0.1, help="Dropout probability for v33 ray-conditioned attention")
    parser.add_argument("--v33_use_ray_bias", action="store_true", default=True, help="Use ray-intersection bias in v33 ray-conditioned attention")
    parser.add_argument("--no_v33_use_ray_bias", dest="v33_use_ray_bias", action="store_false", help="Disable ray-intersection bias in v33 ray-conditioned attention")
    parser.add_argument("--v33_residual_gate_init", type=float, default=-6.0, help="Initial residual gate logit for v33 ray-conditioned attention")
    # v33 hierarchical multi-scale spatial pyramid
    parser.add_argument("--use_hierarchical_multiscale_spatial_pyramid_v33", action="store_true", default=False, help="Use v33 hierarchical multi-scale cross-view spatial pyramid")
    parser.add_argument("--v33_hmsp_scales", type=int, nargs="+", default=[1, 2, 4], help="Joint downsampling factors for v33 spatial pyramid")
    parser.add_argument("--v33_hmsp_n_heads", type=int, default=4, help="Number of attention heads per v33 pyramid scale")
    parser.add_argument("--v33_hmsp_n_part_layers", type=int, default=1, help="Number of layers per v33 pyramid scale")
    parser.add_argument("--v33_hmsp_dropout", type=float, default=0.1, help="Dropout probability for v33 pyramid")
    parser.add_argument("--v33_hmsp_stochastic_depth_prob", type=float, default=0.0, help="Stochastic depth probability for v33 pyramid")
    parser.add_argument("--v33_hmsp_use_geometry_bias", action="store_true", default=True, help="Use geometry bias in v33 pyramid")
    parser.add_argument("--no_v33_hmsp_use_geometry_bias", dest="v33_hmsp_use_geometry_bias", action="store_false", help="Disable geometry bias in v33 pyramid")
    parser.add_argument("--v33_hmsp_use_adaptive_scale_fusion", action="store_true", default=True, help="Use adaptive per-token scale fusion in v33 pyramid")
    parser.add_argument("--no_v33_hmsp_use_adaptive_scale_fusion", dest="v33_hmsp_use_adaptive_scale_fusion", action="store_false", help="Disable adaptive scale fusion in v33 pyramid")
    # v34 view-joint graph network
    parser.add_argument("--use_view_joint_graph_network_v34", action="store_true", default=False, help="Use v34 view-joint graph network")
    parser.add_argument("--v34_vjgn_n_layers", type=int, default=2, help="Number of graph attention layers in v34 VJGN")
    parser.add_argument("--v34_vjgn_n_heads", type=int, default=4, help="Number of attention heads in v34 VJGN")
    parser.add_argument("--v34_vjgn_dropout", type=float, default=0.0, help="Dropout probability for v34 VJGN")
    # v34 geometry-aware view-joint graph network
    parser.add_argument("--use_geometry_view_joint_graph_network_v34", action="store_true", default=False, help="Use v34 geometry-aware view-joint graph network")
    parser.add_argument("--v34_gvjgn_n_layers", type=int, default=2, help="Number of graph attention layers in v34 geometry-aware VJGN")
    parser.add_argument("--v34_gvjgn_n_heads", type=int, default=4, help="Number of attention heads in v34 geometry-aware VJGN")
    parser.add_argument("--v34_gvjgn_dropout", type=float, default=0.0, help="Dropout probability for v34 geometry-aware VJGN")
    # v35 temporal view-joint graph network
    parser.add_argument("--use_temporal_view_joint_graph_network_v35", action="store_true", default=False, help="Use v35 temporal view-joint graph network")
    parser.add_argument("--v35_tvjgn_n_layers", type=int, default=2, help="Number of graph attention layers in v35 temporal VJGN")
    parser.add_argument("--v35_tvjgn_n_heads", type=int, default=4, help="Number of attention heads in v35 temporal VJGN")
    parser.add_argument("--v35_tvjgn_dropout", type=float, default=0.0, help="Dropout probability for v35 temporal VJGN")
    # v36 uncertainty-gated iterative graph refinement
    parser.add_argument("--use_uncertainty_gated_iterative_graph_refinement_v36", action="store_true", default=False, help="Use v36 uncertainty-gated iterative graph refinement")
    parser.add_argument("--v36_ugigr_n_layers", type=int, default=1, help="Number of graph attention layers in v36 UGIGR")
    parser.add_argument("--v36_ugigr_n_iters", type=int, default=2, help="Number of iterative refinement steps in v36 UGIGR")
    parser.add_argument("--v36_ugigr_n_heads", type=int, default=4, help="Number of attention heads in v36 UGIGR")
    parser.add_argument("--v36_ugigr_dropout", type=float, default=0.0, help="Dropout probability for v36 UGIGR")
    parser.add_argument("--v36_ugigr_uncertainty_hidden", type=int, default=64, help="Hidden dimension of v36 UGIGR uncertainty MLP")
    # v43 adaptive per-node residual on top of v36 UGIGR
    parser.add_argument("--use_v43_adaptive_node_residual", action="store_true", default=False, help="Use v43 adaptive per-node residual in v36 UGIGR")
    # v44 edge-type-aware uncertainty gating on top of v36 UGIGR
    parser.add_argument("--use_v44_edge_type_gate", action="store_true", default=False, help="Use v44 edge-type-aware uncertainty gating in v36 UGIGR")
    # v37 self-critique view reliability estimator
    parser.add_argument("--use_self_critique_view_reliability_v37", action="store_true", default=False, help="Use v37 self-critique view reliability estimator")
    parser.add_argument("--v37_scvr_hidden", type=int, default=64, help="Hidden dimension of v37 SCVR MLP")
    parser.add_argument("--v37_scvr_n_layers", type=int, default=2, help="Number of layers in v37 SCVR MLP")
    parser.add_argument("--v37_scvr_use_temporal_context", action="store_true", default=True, help="Use temporal context in v37 SCVR")
    parser.add_argument("--v37_scvr_loss_weight", type=float, default=0.01, help="Weight for v37 SCVR reprojection-residual auxiliary loss")
    # v39 reliability-coupled adaptive graph refinement
    parser.add_argument("--use_reliability_coupled_graph_refinement_v39", action="store_true", default=False, help="Couple v37 reliability scores with v36 uncertainty gates")
    parser.add_argument("--v30_dropout", type=float, default=0.1, help="Dropout for v30 hierarchical encoder")
    parser.add_argument("--v30_stochastic_depth_prob", type=float, default=0.0, help="Stochastic depth probability for v30 hierarchical encoder")
    parser.add_argument("--use_test_time_self_evolution_v29", action="store_true", default=False, help="Use v29 test-time self-evolution with physical-space alignment at inference")
    parser.add_argument("--v29_tte_n_iters", type=int, default=3, help="Number of iterations for v29 test-time self-evolution")
    parser.add_argument("--v29_tte_sigma_reproj", type=float, default=5.0, help="Cauchy kernel scale (pixels) for v29 self-evolution")
    parser.add_argument("--v29_tte_residual_thresh_mm", type=float, default=0.5, help="Early-stop threshold (mm) for v29 self-evolution")
    parser.add_argument("--v29_tte_use_physical_space_alignment", action="store_true", default=True, help="Apply v28 physical-space alignment inside v29 self-evolution")
    parser.add_argument("--no_v29_tte_use_physical_space_alignment", dest="v29_tte_use_physical_space_alignment", action="store_false", help="Disable physical-space alignment inside v29 self-evolution")
    parser.add_argument("--use_physical_space_temporal_loss_v29", action="store_true", default=False, help="Use v29 physical-space temporal loss during training")
    parser.add_argument("--v29_floor_loss_weight", type=float, default=0.01, help="Weight for v29 foot-floor loss")
    parser.add_argument("--v29_bone_temporal_weight", type=float, default=0.01, help="Weight for v29 bone-length temporal loss")
    parser.add_argument("--v29_com_jitter_weight", type=float, default=0.001, help="Weight for v29 center-of-mass jitter loss")
    parser.add_argument("--v29_physical_loss_warmup_epochs", type=int, default=0, help="Linear warmup epochs for v29 physical loss weights (0=disabled)")
    parser.add_argument("--use_physical_collision_penalty_v31", action="store_true", default=False, help="Use v31 self-collision penalty on predicted 3D poses")
    parser.add_argument("--v31_collision_loss_weight", type=float, default=0.001, help="Weight for v31 collision penalty")
    parser.add_argument("--v31_collision_margin", type=float, default=0.05, help="Safety margin for v31 collision penalty")
    parser.add_argument("--v31_collision_warmup_epochs", type=int, default=0, help="Linear warmup epochs for v31 collision penalty")
    parser.add_argument("--kap_use_angle_limit", action="store_true", default=True, help="Use soft joint-angle limit penalty in v22 KAP")
    parser.add_argument("--kap_max_flexion_deg", type=float, default=160.0, help="Maximum flexion for v22 joint-angle limit")
    parser.add_argument("--use_full_precision_dlt", action="store_true", help="Use full 2x2 precision matrix in DLT triangulation")
    parser.add_argument("--use_robust_dlt_reweight", action="store_true", help="Robust reweighting pass inside full-precision DLT triangulation")
    parser.add_argument("--use_irls_reweight", action="store_true", help="IRLS Cauchy robust reweighting after one-step robust DLT")
    parser.add_argument("--irls_n_iters", type=int, default=2, help="Number of IRLS iterations")
    parser.add_argument("--irls_cauchy_scale", type=float, default=1.0, help="Cauchy kernel scale for IRLS reweighting")
    parser.add_argument("--use_domain_embedding", action="store_true", help="Add a learnable per-dataset embedding (requires --use_mixed_loader)")
    parser.add_argument("--num_domains", type=int, default=2, help="Number of dataset domains for the embedding")
    parser.add_argument("--domain_loss_weights", type=str, default=None, help="Comma-separated per-domain MSE weights for mixed training (e.g. '1.0,1.5')")
    parser.add_argument("--monotonic_loss_weight", type=float, default=0.0, help="Weight for monotonic multi-view ranking loss")
    parser.add_argument("--monotonic_margin", type=float, default=5.0, help="Margin (mm) for monotonic multi-view loss")
    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=13, help="Temporal clip length")
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--val_stride", type=int, default=1, help="Validation clip stride")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_cosine", action="store_true", help="Use cosine LR schedule with warmup")
    parser.add_argument("--lr_warmup_epochs", type=int, default=0, help="Warmup epochs (requires --lr_cosine)")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum LR for cosine schedule")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Global gradient clipping norm")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Augmentation
    parser.add_argument("--noise_std", type=float, default=0.5, help="2-D observation noise std (pixels)")
    parser.add_argument("--confidence_dropout", type=float, default=0.0, help="Confidence dropout rate")
    parser.add_argument("--view_dropout_rate", type=float, default=0.0, help="Probability of dropping a full view")
    parser.add_argument("--min_views", type=int, default=2, help="Minimum kept views during view dropout")
    parser.add_argument("--outlier_view_prob", type=float, default=0.0, help="Probability of injecting outlier views per clip (0 disables)")
    parser.add_argument("--outlier_view_max_views", type=int, default=1, help="Maximum number of views to corrupt when injecting outliers")
    parser.add_argument("--outlier_view_offset_std", type=float, default=10.0, help="Std dev of large per-view 2-D offset for outlier views (pixels)")
    parser.add_argument("--outlier_view_noise_std", type=float, default=15.0, help="Std dev of per-pixel Gaussian noise for outlier views (pixels)")
    parser.add_argument("--occlusion_augment_prob", type=float, default=0.0, help="Per-clip probability of applying joint-level occlusion/noise augmentation (0 disables)")
    parser.add_argument("--occlusion_joint_rate", type=float, default=0.1, help="Probability of occluding each (view, joint) pair")
    parser.add_argument("--occlusion_multiview_rate", type=float, default=0.05, help="Fraction of joints to occlude across multiple views")
    parser.add_argument("--occlusion_temporal_prob", type=float, default=0.1, help="Probability of temporal joint occlusion streak")
    parser.add_argument("--occlusion_per_joint_noise_std", type=float, default=1.5, help="Std of per-joint 2-D noise (pixels)")
    parser.add_argument("--occlusion_per_view_noise_std", type=float, default=0.5, help="Std of per-view shared 2-D noise (pixels)")
    parser.add_argument("--occlusion_min_visible_joints", type=int, default=5, help="Minimum visible joints per view after occlusion")
    parser.add_argument("--variable_view_subset", action="store_true", help="Train with random view-subset sampling (k ~ Uniform(min_views, V))")
    parser.add_argument("--use_variable_view_training", action="store_true", help="Randomly sample and permute view subsets each batch")
    parser.add_argument("--variable_view_min_views", type=int, default=2, help="Minimum views in variable-view training subset")
    parser.add_argument("--variable_view_max_views", type=int, default=4, help="Maximum views in variable-view training subset")
    parser.add_argument("--variable_view_permute", action="store_true", help="Permute selected view order during variable-view training")
    parser.add_argument("--variable_view_curriculum_alpha", type=float, default=1.0, help="Curriculum exponent for variable-view maximum (1.0=linear)")
    parser.add_argument("--variable_view_max_views_start", type=int, default=None, help="Maximum views at epoch 0 (curriculum start); defaults to variable_view_max_views")
    parser.add_argument("--domain_aware_view_curriculum", action="store_true", help="Clamp variable-view subset to the real number of cameras per domain (H36M=4, MPI=14)")
    # Calibration curriculum
    parser.add_argument("--cam_aug_schedule", type=str, default="none", choices=["none", "extended_curriculum", "extended_intrinsics_curriculum"], help="Camera perturbation schedule")
    parser.add_argument("--cam_aug_rot", type=float, default=0.5, help="Rotation perturbation std (degrees)")
    parser.add_argument("--cam_aug_trans", type=float, default=0.005, help="Translation perturbation std")
    parser.add_argument("--cam_aug_focal", type=float, default=0.01, help="Focal-length perturbation std (fraction)")
    parser.add_argument("--cam_aug_pp", type=float, default=2.0, help="Principal-point perturbation std (pixels)")
    parser.add_argument("--cam_aug_ramp_epochs", type=int, default=10, help="Epochs to ramp extrinsics")
    parser.add_argument("--cam_aug_intrinsics_ramp_epochs", type=int, default=5, help="Epochs to ramp intrinsics")
    parser.add_argument("--cam_aug_warmup_epochs", type=int, default=0, help="Warmup epochs before ramping")
    # Loss weights
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Visibility BCE weight")
    parser.add_argument("--pa_loss_weight", type=float, default=0.0, help="Weight for Procrustes-aligned MSE loss (0 disables)")
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.05, help="Uncertainty NLL weight")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.02, help="Temporal consistency weight")
    parser.add_argument("--temporal_acceleration_weight", type=float, default=0.0,
                        help="Relative weight of acceleration smoothness within the temporal loss (0 disables)")
    parser.add_argument("--bone_loss_weight", type=float, default=0.05, help="Bone-length consistency weight")
    parser.add_argument("--joint_limit_weight", type=float, default=0.0, help="Joint-limit (hyper-extension) auxiliary loss weight")
    parser.add_argument("--joint_limit_max_flexion", type=float, default=160.0, help="Maximum allowed interior joint angle in degrees")
    parser.add_argument("--temporal_bone_weight", type=float, default=0.0, help="Temporal bone-length consistency weight")
    # v40 skeleton-aware physical loss
    parser.add_argument("--use_skeleton_physical_loss_v40", action="store_true", default=False, help="Use v40 composite skeleton-aware physical loss")
    parser.add_argument("--v40_bone_weight", type=float, default=0.05, help="Bone-length matching weight in v40 physical loss")
    parser.add_argument("--v40_joint_limit_weight", type=float, default=0.01, help="Joint-limit penalty weight in v40 physical loss")
    parser.add_argument("--v40_symmetry_weight", type=float, default=0.02, help="Left/right symmetry weight in v40 physical loss")
    parser.add_argument("--v40_floor_weight", type=float, default=0.02, help="Foot-floor contact weight in v40 physical loss")
    parser.add_argument("--v40_warmup_epochs", type=int, default=0, help="Linearly ramp v40 physical loss over this many epochs (0 disables)")
    parser.add_argument("--attention_entropy_weight", type=float, default=0.0, help="Attention-entropy regularisation weight")
    parser.add_argument("--budget_loss_weight", type=float, default=0.0, help="Adaptive-view budget loss weight")
    parser.add_argument("--reproj_loss_weight", type=float, default=0.0, help="2D reprojection loss weight")
    parser.add_argument("--aleatoric_reproj_loss_weight", type=float, default=0.0, help="Aleatoric 2D reprojection NLL weight")
    parser.add_argument("--reproj_warmup_epochs", type=int, default=0, help="Linearly ramp reprojection loss weights from 0 over this many epochs (0 disables)")
    # Warm-start
    parser.add_argument("--warm_start", type=str, default=None, help="Path to v2/v3 checkpoint for warm-starting")
    parser.add_argument("--warm_start_freeze_epochs", type=int, default=0, help="Freeze encoder/transformer for N epochs after warm-start")
    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Stop training if val_loss does not improve for N epochs (0 disables)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum val_loss improvement to reset early stopping counter")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay for Adam optimizer (0 disables)")
    # I/O
    parser.add_argument("--output", type=str, default="outputs/omniview_fusion_v5_webbridge_multi.pth", help="Checkpoint path")
    args = parser.parse_args()

    # Parse v48 comma-separated 3DPW actual-mode validation paths.
    if getattr(args, "v48_3dpw_actual_val_paths", None) is not None:
        args.v48_3dpw_actual_val_paths = [
            p.strip() for p in args.v48_3dpw_actual_val_paths.split(",") if p.strip()
        ]
    else:
        args.v48_3dpw_actual_val_paths = None

    # Parse v48 per-domain view-dropout dictionary (JSON string).
    if getattr(args, "v48_dropout_per_domain", None) is not None:
        import json
        try:
            parsed = json.loads(args.v48_dropout_per_domain)
            args.v48_dropout_per_domain = {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            raise ValueError(
                f"--v48_dropout_per_domain must be a JSON dict, got: {args.v48_dropout_per_domain}"
            )

    # v48 needs at least 6 domains (h36m, mpi, aist, shelf, campus, 3dpw).
    if getattr(args, "use_v48_domain_generalization", False):
        args.num_domains = max(args.num_domains, 6)

    # Default manifest if none provided and no legacy --train/--val.
    if args.manifest is None and (args.train is None or args.val is None):
        args.manifest = ["configs/splits/webbridge_all_train.yaml"]

    if args.smoke:
        args.epochs = 1
        args.d = 32
        args.residual_hidden = 64
        args.n_st_layers = 1
        args.graph_num_layers = 1
        args.batch_size = 2
        args.clip_len = 9
        args.train_samples = 16
        args.val_stride = 1

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Windows/WSL cannot use PyTorch DataLoader workers reliably; force 0.
    if os.name == "nt" and args.num_workers != 0:
        warnings.warn(
            f"Windows detected: num_workers={args.num_workers} is unsupported; "
            "falling back to num_workers=0."
        )
        args.num_workers = 0

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset, val_dataset, n_views, n_joints = build_datasets(args)

    selected_collate_fn = webbridge_mixed_collate_fn if args.use_mixed_loader else collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=selected_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=selected_collate_fn,
        num_workers=args.num_workers,
    )

    args.j = n_joints
    args.n_views = n_views

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model_from_args(args, n_joints, n_views, device=device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------
    if args.warm_start is not None:
        load_warm_start(model, args.warm_start)

    # ------------------------------------------------------------------
    # Optimizer / scheduler / trainer
    # ------------------------------------------------------------------
    trainable_params = list(model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_epochs = args.epochs
    warmup_epochs = args.lr_warmup_epochs if args.lr_cosine else 0
    scheduler = build_lr_scheduler(optimizer, total_epochs, warmup_epochs, args.lr_min) if args.lr_cosine else None

    ema_decay = args.ema_decay if args.ema_decay > 0.0 else None

    trainer = OmniMultiViewTrainer(
        model,
        optimizer,
        device,
        args,
        total_epochs=total_epochs,
        max_grad_norm=args.max_grad_norm,
        amp_enabled=args.amp,
        ema_decay=ema_decay,
        scheduler=scheduler,
    )

    # ------------------------------------------------------------------
    # Optional staged warm-start / v47 temporal head warm-up.
    # ------------------------------------------------------------------
    freeze_epochs = args.warm_start_freeze_epochs if args.warm_start else 0
    v47_freeze_epochs = args.v47_freeze_base_epochs if args.use_v47_temporal_aggregation else 0
    if freeze_epochs > 0 and freeze_epochs >= total_epochs:
        print("warm_start_freeze_epochs >= total_epochs; no end-to-end training will occur.")
        freeze_epochs = max(0, total_epochs - 1)
    if v47_freeze_epochs > 0 and v47_freeze_epochs >= total_epochs:
        print("v47_freeze_base_epochs >= total_epochs; no end-to-end training will occur.")
        v47_freeze_epochs = max(0, total_epochs - 1)
    total_freeze_epochs = freeze_epochs + v47_freeze_epochs

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the training configuration immediately so that even partial runs can be
    # evaluated / resumed without guessing the exact hyper-parameters.
    try:
        import json

        config_path = output_path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    except Exception as exc:
        print(f"Warning: could not save training config: {exc}")

    if freeze_epochs > 0:
        print(f"Freezing encoder / ST-transformer for {freeze_epochs} epoch(s)...")
        freeze_old_params(model)
        for freeze_i in range(freeze_epochs):
            trainer.epoch += 1
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader, compute_metric=build_eval_metric())
            trainer.step_scheduler()
            print(
                f"[Freeze] Epoch {trainer.epoch}: train_loss={train_metrics.get('loss', float('nan')):.6f}, "
                f"val_MPJPE={val_metrics.get('mpjpe', float('nan')) * 1000:.2f}mm"
            )
        print("Unfreezing full model for end-to-end training.")
        unfreeze_all(model)

    if v47_freeze_epochs > 0:
        print(f"Freezing base model for {v47_freeze_epochs} epoch(s) to warm-start v47 temporal head...")
        freeze_v47_base_params(model)
        for freeze_i in range(v47_freeze_epochs):
            trainer.epoch += 1
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader, compute_metric=build_eval_metric())
            trainer.step_scheduler()
            print(
                f"[v47 Freeze] Epoch {trainer.epoch}: train_loss={train_metrics.get('loss', float('nan')):.6f}, "
                f"val_MPJPE={val_metrics.get('mpjpe', float('nan')) * 1000:.2f}mm"
            )
        print("Unfreezing full model for end-to-end v47 training.")
        unfreeze_all(model)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    remaining_epochs = total_epochs - total_freeze_epochs
    if remaining_epochs > 0:
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=remaining_epochs,
            eval_metric=build_eval_metric(),
            checkpoint_path=str(output_path),
            save_best=True,
            log_interval=50,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )

        best = min(history, key=lambda e: e.get("val", {}).get("loss", float("inf")))
        best_mpjpe = best.get("val", {}).get("mpjpe", float("nan"))
        print(f"Best val MPJPE: {best_mpjpe * 1000:.2f}mm -> {output_path}")

    # Always save the final checkpoint as well.
    final_path = str(output_path.with_suffix("")) + "_final.pth"
    trainer.save_checkpoint(final_path)

    # Save the training configuration so the eval script can reconstruct the model
    # with the exact flags used during training.
    config_path = output_path.with_suffix(".config.json")
    try:
        import json

        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved training config -> {config_path}")
    except Exception as exc:
        print(f"Warning: could not save training config: {exc}")


if __name__ == "__main__":
    main()
