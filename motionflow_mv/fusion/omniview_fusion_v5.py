"""OmniMultiViewFusion v5 — camera-conditioned, set-transformer multi-view fusion.

OmniMultiViewFusionV5 subclasses :class:`OmniMultiViewFusionV4` and addresses the
fixed-view-index limitation of the learned ``view_pos_embed`` embedding.

New toggles
-----------
* ``use_camera_view_embedding`` – replace the learned view positional embedding
  with an MLP conditioned on calibrated camera intrinsics and extrinsics.
* ``use_set_view_aggregator`` – add a permutation-invariant set-transformer
  (Induced Set Attention Blocks) over views before the time+view transformer.
* ``use_diffusion_refiner_v20`` – replace the deterministic residual MLP with a
  lightweight diffusion-based pose refiner.

The model also accepts an explicit ``view_mask`` so that missing views can be
masked out in confidences, weights, and triangulation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from motionflow_mv.fusion.camera_conditioned_view_embedding import (
    CameraConditionedViewEmbedding,
)
from motionflow_mv.fusion.epipolar_transformer_bias import (
    EpipolarBiasedTransformerEncoderLayer,
)
from motionflow_mv.fusion.cross_view_transformer_v17 import CrossViewTransformerV17
from motionflow_mv.fusion.deformable_cross_view_attention import DeformableCrossViewAttention
from motionflow_mv.fusion.diffusion_pose_refiner_v20 import DiffusionPoseRefinerV20
from motionflow_mv.fusion.kinematic_anthropometric_prior_v22 import (
    KinematicAnthropometricPrior,
)
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import MultiViewGeometryFusionV25
from motionflow_mv.fusion.temporal_geometry_fusion_v26 import TemporalGeometryFusionV26
from motionflow_mv.fusion.test_time_self_evolution_v27 import TestTimeSelfEvolutionV27
from motionflow_mv.fusion.physical_space_alignment_v28 import (
    PhysicalSpaceAlignmentV28,
    floor_loss,
    bone_temporal_loss,
)
from motionflow_mv.fusion.self_evolving_hierarchical_multiview_v29 import (
    HierarchicalViewEncoderV29,
    PhysicalSpaceTemporalLossV29,
    TestTimeSelfEvolutionV29,
)
from motionflow_mv.fusion.hierarchical_multiview_v30 import HierarchicalViewEncoderV30
from motionflow_mv.fusion.hierarchical_multiview_v31 import HierarchicalViewEncoderV31
from motionflow_mv.fusion.camera_view_embedding_v31 import CameraConditionedViewEmbeddingV31
from motionflow_mv.losses.physical_collision_penalty_v31 import PhysicalCollisionPenaltyV31
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)
from motionflow_mv.fusion.neural_bundle_adjustment_v21 import NeuralBundleAdjustment
from motionflow_mv.fusion.omniview_fusion_v4 import OmniMultiViewFusionV4
from motionflow_mv.fusion.perceiver_view_aggregator import PerceiverViewAggregator
from motionflow_mv.fusion.temporal_perceiver_v19 import TemporalPerceiverRefiner
from motionflow_mv.fusion.variable_view_set_aggregator import (
    VariableViewSetAggregator,
)


class OmniMultiViewFusionV5(OmniMultiViewFusionV4):
    """OmniMultiViewFusion v5 prototype.

    Parameters
    ----------
    use_camera_view_embedding:
        Replace the learned view positional embedding with a camera-conditioned
        MLP of ``(K, R, t)``.
    use_set_view_aggregator:
        Apply a permutation-invariant set-transformer aggregator over views
        before the time+view transformer.
    camera_view_embedding_hidden:
        Hidden dimension of the camera-conditioned view embedding MLP.
    set_view_n_isab_layers:
        Number of ISAB layers in the set aggregator.
    set_view_num_inducing_points:
        Number of inducing points in each ISAB.
    set_view_dropout:
        Dropout probability in the set aggregator attention layers.
    use_diffusion_refiner_v20:
        Replace the deterministic residual MLP with a diffusion-based refiner.
    num_diffusion_steps:
        Number of diffusion timesteps used by the v20 refiner.
    See ``OmniMultiViewFusionV4`` for the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 0,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_covariance: bool = True,
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
        graph_num_layers: int = 1,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        graph_dropout: float = 0.0,
        use_multiscale_fusion: bool = True,
        use_adaptive_multiscale_fusion: bool = False,
        use_camera_conditioning: bool = True,
        use_epipolar_bias: bool = True,
        multiscale_scales: List[int] = (1, 2, 4),  # type: ignore[assignment]
        camera_condition_dim: int = 32,
        epipolar_temperature: float = 10.0,
        use_context_visibility: bool = False,
        use_skeleton_residual: bool = False,
        use_skeleton_residual_v31: bool = False,
        use_kinematic_refiner: bool = False,
        use_adaptive_view_selection: bool = False,
        use_rotation_correction: bool = False,
        use_entropy_regularization: bool = False,
        adaptive_view_target_k: int = 2,
        rotation_max_rot_deg: float = 2.0,
        entropy_weight: float = 0.01,
        use_camera_view_embedding: bool = False,
        use_camera_view_embedding_v31: bool = False,
        use_set_view_aggregator: bool = False,
        use_perceiver_aggregator: bool = False,
        use_cross_view_transformer_v17: bool = False,
        use_deformable_cross_view_attention_v18: bool = False,
        deformable_attention_use_topk_st: bool = False,
        use_temporal_perceiver_v19: bool = False,
        use_diffusion_refiner_v20: bool = False,
        use_neural_bundle_adjustment_v21: bool = False,
        use_kinematic_anthropometric_prior_v22: bool = False,
        use_multiview_geometry_fusion_v25: bool = False,
        v25_use_geometry_attention: bool = True,
        v25_use_learned_depth_triangulation: bool = True,
        v25_use_geometry_bundle_adjustment: bool = True,
        v25_use_camera_joint_graph: bool = False,
        v25_use_outlier_view_detector: bool = False,
        v25_outlier_z_thresh: float = 3.0,
        v25_outlier_soft_beta: float = 1.0,
        v25_geom_loss_weight: float = 0.1,
        v25_dropout: float = 0.1,
        use_temporal_geometry_fusion_v26: bool = False,
        v26_temporal_window: int = 3,
        v26_temporal_attention_residual_gate_init: float = 0.0,
        use_uncertainty_depth_proposals_v27: bool = False,
        v27_uncertainty_loss_weight: float = 0.01,
        v27_udp_n_mixtures: int = 1,
        use_test_time_self_evolution_v27: bool = False,
        v27_tte_n_iters: int = 3,
        use_physical_space_alignment_v28: bool = False,
        use_physical_space_alignment_v32: bool = False,
        v28_floor_loss_weight: float = 0.0,
        v28_bone_temporal_weight: float = 0.0,
        v27_tte_sigma_reproj: float = 5.0,
        v27_tte_residual_thresh_mm: float = 0.5,
        # v29 toggles
        use_hierarchical_multiview_v29: bool = False,
        v29_n_heads: int = 4,
        v29_n_part_layers: int = 1,
        use_test_time_self_evolution_v29: bool = False,
        v29_tte_n_iters: int = 3,
        v29_tte_sigma_reproj: float = 5.0,
        v29_tte_residual_thresh_mm: float = 0.5,
        v29_tte_use_physical_space_alignment: bool = True,
        use_physical_space_temporal_loss_v29: bool = False,
        v29_floor_loss_weight: float = 0.01,
        v29_bone_temporal_weight: float = 0.01,
        v29_com_jitter_weight: float = 0.001,
        v29_physical_loss_warmup_epochs: int = 0,
        # v31 physical collision penalty
        use_physical_collision_penalty_v31: bool = False,
        v31_collision_loss_weight: float = 0.001,
        v31_collision_margin: float = 0.05,
        v31_collision_warmup_epochs: int = 0,
        # v30 toggles
        use_hierarchical_multiview_v30: bool = False,
        v30_n_heads: int = 4,
        v30_n_part_layers: int = 1,
        v30_dropout: float = 0.1,
        v30_stochastic_depth_prob: float = 0.0,
        use_hierarchical_multiview_v31: bool = False,
        v31_geometry_bias: bool = True,
        v31_use_ray_attention: bool = False,
        # v32 temporal trajectory consistency
        use_trajectory_consistency_v32: bool = False,
        v32_smooth_weight: float = 1e-3,
        v32_drift_weight: float = 1e-2,
        kap_loss_weight: float = 0.01,
        kap_use_angle_limit: bool = True,
        kap_max_flexion_deg: float = 160.0,
        kap_max_delta: float = 0.10,
        num_diffusion_steps: int = 10,
        camera_view_embedding_hidden: int = 32,
        set_view_n_isab_layers: int = 2,
        set_view_num_inducing_points: int = 32,
        set_view_dropout: float = 0.0,
        perceiver_n_latents: int = 16,
        perceiver_n_layers: int = 2,
        perceiver_n_heads: int = 4,
        perceiver_dropout: float = 0.0,
        use_full_precision_dlt: bool = False,
        use_robust_dlt_reweight: bool = False,
        use_domain_embedding: bool = False,
        use_irls_reweight: bool = False,
        irls_n_iters: int = 2,
        irls_cauchy_scale: float = 1.0,
        num_domains: int = 2,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_covariance=return_covariance,
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
            graph_num_layers=graph_num_layers,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
            graph_dropout=graph_dropout,
            use_multiscale_fusion=use_multiscale_fusion,
            use_camera_conditioning=use_camera_conditioning,
            use_epipolar_bias=use_epipolar_bias,
            multiscale_scales=multiscale_scales,
            camera_condition_dim=camera_condition_dim,
            epipolar_temperature=epipolar_temperature,
            use_context_visibility=use_context_visibility,
            use_skeleton_residual=use_skeleton_residual,
            use_skeleton_residual_v31=use_skeleton_residual_v31,
            use_kinematic_refiner=use_kinematic_refiner,
            use_adaptive_view_selection=use_adaptive_view_selection,
            use_rotation_correction=use_rotation_correction,
            use_entropy_regularization=use_entropy_regularization,
            adaptive_view_target_k=adaptive_view_target_k,
            rotation_max_rot_deg=rotation_max_rot_deg,
            entropy_weight=entropy_weight,
        )

        self.max_temporal_len = max_temporal_len

        # Optional v32 temporal trajectory consistency.
        self.use_trajectory_consistency_v32 = use_trajectory_consistency_v32
        self.v32_smooth_weight = v32_smooth_weight
        self.v32_drift_weight = v32_drift_weight
        if self.use_trajectory_consistency_v32:
            from motionflow_mv.fusion.trajectory_consistency_v32 import (
                TrajectoryConsistencyRefinerV32,
            )
            self.trajectory_consistency_refiner = TrajectoryConsistencyRefinerV32(j)
        else:
            self.trajectory_consistency_refiner = None

        # Optional adaptive scale-selective multi-scale fusion (v12).
        self.use_adaptive_multiscale_fusion = use_adaptive_multiscale_fusion
        if use_adaptive_multiscale_fusion and self.multiscale_fusion is not None:
            from motionflow_mv.fusion.adaptive_hierarchical_multiscale_fusion import (
                AdaptiveHierarchicalMultiscaleFusion,
            )
            self.multiscale_fusion = AdaptiveHierarchicalMultiscaleFusion(
                d=d,
                n_views=n_views,
                scales=multiscale_scales,
                n_heads=n_heads,
                dropout=0.1,
            )

        self.use_camera_view_embedding = use_camera_view_embedding
        self.use_camera_view_embedding_v31 = use_camera_view_embedding_v31
        self.use_set_view_aggregator = use_set_view_aggregator
        self.use_perceiver_aggregator = use_perceiver_aggregator
        self.camera_view_embedding_hidden = camera_view_embedding_hidden
        self.set_view_n_isab_layers = set_view_n_isab_layers
        self.set_view_num_inducing_points = set_view_num_inducing_points
        self.set_view_dropout = set_view_dropout
        self.perceiver_n_latents = perceiver_n_latents
        self.perceiver_n_layers = perceiver_n_layers
        self.perceiver_n_heads = perceiver_n_heads
        self.perceiver_dropout = perceiver_dropout
        self.use_full_precision_dlt = use_full_precision_dlt
        self.use_robust_dlt_reweight = use_robust_dlt_reweight
        self.use_irls_reweight = use_irls_reweight
        self.irls_n_iters = irls_n_iters
        self.irls_cauchy_scale = irls_cauchy_scale
        self.use_domain_embedding = use_domain_embedding
        if self.use_domain_embedding:
            self.domain_embedding = nn.Embedding(num_domains, d)

        if self.use_camera_view_embedding_v31:
            self.camera_view_embedding = CameraConditionedViewEmbeddingV31(
                d=d,
                camera_hidden=camera_view_embedding_hidden,
            )
        elif self.use_camera_view_embedding:
            self.camera_view_embedding = CameraConditionedViewEmbedding(
                d=d,
                camera_hidden=camera_view_embedding_hidden,
            )
        else:
            self.camera_view_embedding = None

        if self.use_set_view_aggregator:
            self.set_view_aggregator = VariableViewSetAggregator(
                d=d,
                n_heads=n_heads,
                n_isab_layers=set_view_n_isab_layers,
                num_inducing_points=set_view_num_inducing_points,
                dropout=set_view_dropout,
            )
        else:
            self.set_view_aggregator = None

        if self.use_perceiver_aggregator:
            self.perceiver_aggregator = PerceiverViewAggregator(
                d=d,
                n_heads=perceiver_n_heads,
                n_latents=perceiver_n_latents,
                n_layers=perceiver_n_layers,
                dropout=perceiver_dropout,
            )
        else:
            self.perceiver_aggregator = None

        self.use_cross_view_transformer_v17 = use_cross_view_transformer_v17
        if self.use_cross_view_transformer_v17:
            self.cross_view_transformer_v17 = CrossViewTransformerV17(
                d=d,
                n_heads=n_heads,
                n_layers=2,
                dropout=0.1,
            )
        else:
            self.cross_view_transformer_v17 = None

        self.use_deformable_cross_view_attention_v18 = use_deformable_cross_view_attention_v18
        self.deformable_attention_use_topk_st = deformable_attention_use_topk_st
        if self.use_deformable_cross_view_attention_v18:
            self.deformable_cross_view_attention_v18 = DeformableCrossViewAttention(
                d=d,
                n_heads=n_heads,
                n_views=n_views,
                n_samples=max(2, n_views // 2),
                epipolar_temperature=10.0,
                dropout=0.1,
                use_topk_straight_through=deformable_attention_use_topk_st,
            )
        else:
            self.deformable_cross_view_attention_v18 = None

        self.use_temporal_perceiver_v19 = use_temporal_perceiver_v19
        if self.use_temporal_perceiver_v19:
            self.temporal_perceiver_refiner_v19 = TemporalPerceiverRefiner(
                j=self.j,
                in_dim=3 + self.d,
                d=64,
                n_latents=32,
                n_layers=2,
                n_heads=4,
                dropout=0.0,
                max_temporal_len=self.max_temporal_len,
            )
        else:
            self.temporal_perceiver_refiner_v19 = None

        self.use_diffusion_refiner_v20 = use_diffusion_refiner_v20
        self.num_diffusion_steps = num_diffusion_steps
        if self.use_diffusion_refiner_v20:
            self.diffusion_refiner_v20 = DiffusionPoseRefinerV20(
                j=self.j,
                in_dim=self.d,
                residual_hidden=256,
                num_diffusion_steps=num_diffusion_steps,
            )
        else:
            self.diffusion_refiner_v20 = None

        self.use_neural_bundle_adjustment_v21 = use_neural_bundle_adjustment_v21
        if self.use_neural_bundle_adjustment_v21:
            self.neural_bundle_adjustment_v21 = NeuralBundleAdjustment(
                n_iters=3,
                camera_hidden=256,
            )
        else:
            self.neural_bundle_adjustment_v21 = None

        # Optional v22 kinematic anthropometric prior.
        self.use_kinematic_anthropometric_prior_v22 = use_kinematic_anthropometric_prior_v22
        self.kap_loss_weight = kap_loss_weight
        if self.use_kinematic_anthropometric_prior_v22:
            self.kinematic_anthropometric_prior_v22 = KinematicAnthropometricPrior(
                j=self.j,
                d=self.d,
                use_angle_limit=kap_use_angle_limit,
                max_flexion_deg=kap_max_flexion_deg,
                max_delta=kap_max_delta,
            )
        else:
            self.kinematic_anthropometric_prior_v22 = None

        # Optional v25 multi-view geometry fusion.
        self.use_multiview_geometry_fusion_v25 = use_multiview_geometry_fusion_v25
        self.use_temporal_geometry_fusion_v26 = use_temporal_geometry_fusion_v26
        self.v25_geom_loss_weight = v25_geom_loss_weight
        if self.use_temporal_geometry_fusion_v26:
            self.multiview_geometry_fusion_v25 = TemporalGeometryFusionV26(
                d=self.d,
                n_heads=self.n_heads,
                n_views=n_views,
                temporal_window=v26_temporal_window,
                temporal_attention_residual_gate_init=v26_temporal_attention_residual_gate_init,
                use_geometry_attention=v25_use_geometry_attention,
                use_learned_depth_triangulation=v25_use_learned_depth_triangulation,
                use_uncertainty_depth_proposals_v27=use_uncertainty_depth_proposals_v27,
                v27_uncertainty_loss_weight=v27_uncertainty_loss_weight,
                v27_udp_n_mixtures=v27_udp_n_mixtures,
            )
        elif self.use_multiview_geometry_fusion_v25:
            self.multiview_geometry_fusion_v25 = MultiViewGeometryFusionV25(
                d=self.d,
                n_heads=self.n_heads,
                n_views=n_views,
                use_geometry_attention=v25_use_geometry_attention,
                use_learned_depth_triangulation=v25_use_learned_depth_triangulation,
                use_geometry_bundle_adjustment=v25_use_geometry_bundle_adjustment,
                use_camera_joint_graph=v25_use_camera_joint_graph,
                use_outlier_view_detector=v25_use_outlier_view_detector,
                outlier_z_thresh=v25_outlier_z_thresh,
                outlier_soft_beta=v25_outlier_soft_beta,
                dropout=v25_dropout,
                use_uncertainty_depth_proposals_v27=use_uncertainty_depth_proposals_v27,
                v27_uncertainty_loss_weight=v27_uncertainty_loss_weight,
                v27_udp_n_mixtures=v27_udp_n_mixtures,
            )
        else:
            self.multiview_geometry_fusion_v25 = None

        # Optional v27 test-time self-evolution (inference only).
        self.use_test_time_self_evolution_v27 = use_test_time_self_evolution_v27
        self.use_physical_space_alignment_v28 = use_physical_space_alignment_v28 or use_physical_space_alignment_v32
        self.use_physical_space_alignment_v32 = use_physical_space_alignment_v32
        self.v28_floor_loss_weight = v28_floor_loss_weight
        self.v28_bone_temporal_weight = v28_bone_temporal_weight
        if self.use_physical_space_alignment_v28:
            self.physical_space_alignment_v28 = PhysicalSpaceAlignmentV28(
                j=self.j, use_v32=use_physical_space_alignment_v32
            )
        else:
            self.physical_space_alignment_v28 = None
        if self.use_test_time_self_evolution_v27:
            self.test_time_self_evolution_v27 = TestTimeSelfEvolutionV27(
                n_iters=v27_tte_n_iters,
                sigma_reproj=v27_tte_sigma_reproj,
                residual_thresh_mm=v27_tte_residual_thresh_mm,
            )
        else:
            self.test_time_self_evolution_v27 = None

        # Optional v29/v30/v31 hierarchical multi-view encoder.
        self.use_hierarchical_multiview_v29 = use_hierarchical_multiview_v29
        self.use_hierarchical_multiview_v30 = use_hierarchical_multiview_v30
        self.use_hierarchical_multiview_v31 = use_hierarchical_multiview_v31
        self.v31_geometry_bias = v31_geometry_bias
        self.v31_use_ray_attention = v31_use_ray_attention
        if self.use_hierarchical_multiview_v31:
            self.hierarchical_multiview_v31 = HierarchicalViewEncoderV31(
                d=self.d,
                n_heads=v30_n_heads,
                n_part_layers=v30_n_part_layers,
                dropout=v30_dropout,
                stochastic_depth_prob=v30_stochastic_depth_prob,
                use_ray_attention=v31_use_ray_attention,
            )
            self.hierarchical_multiview_v29 = None
            self.hierarchical_multiview_v30 = None
        elif self.use_hierarchical_multiview_v30:
            self.hierarchical_multiview_v30 = HierarchicalViewEncoderV30(
                d=self.d,
                n_heads=v30_n_heads,
                n_views=n_views,
                n_part_layers=v30_n_part_layers,
                dropout=v30_dropout,
                stochastic_depth_prob=v30_stochastic_depth_prob,
            )
            self.hierarchical_multiview_v29 = None
        elif self.use_hierarchical_multiview_v29:
            self.hierarchical_multiview_v29 = HierarchicalViewEncoderV29(
                d=self.d,
                n_heads=v29_n_heads,
                n_views=n_views,
                n_part_layers=v29_n_part_layers,
                dropout=0.1,
            )
            self.hierarchical_multiview_v30 = None
        else:
            self.hierarchical_multiview_v29 = None
            self.hierarchical_multiview_v30 = None

        # Optional v29 test-time self-evolution (with physical-space alignment).
        self.use_test_time_self_evolution_v29 = use_test_time_self_evolution_v29
        if self.use_test_time_self_evolution_v29:
            self.test_time_self_evolution_v29 = TestTimeSelfEvolutionV29(
                n_iters=v29_tte_n_iters,
                sigma_reproj=v29_tte_sigma_reproj,
                residual_thresh_mm=v29_tte_residual_thresh_mm,
                use_physical_space_alignment=v29_tte_use_physical_space_alignment,
                max_residual=0.05,
                j=self.j,
            )
        else:
            self.test_time_self_evolution_v29 = None

        # Optional v29 physical-space temporal loss (training only).
        self.use_physical_space_temporal_loss_v29 = use_physical_space_temporal_loss_v29
        self.v29_floor_loss_weight = v29_floor_loss_weight
        self.v29_bone_temporal_weight = v29_bone_temporal_weight
        self.v29_com_jitter_weight = v29_com_jitter_weight
        self.v29_physical_loss_warmup_epochs = v29_physical_loss_warmup_epochs
        if self.use_physical_space_temporal_loss_v29:
            parents = None
            if self.j == 17:
                parents = H36M_17_PARENTS
            elif self.j == 28:
                parents = MPI_INF_3DHP_28_PARENTS
            self.physical_space_temporal_loss_v29 = PhysicalSpaceTemporalLossV29(
                floor_loss_weight=v29_floor_loss_weight,
                bone_temporal_weight=v29_bone_temporal_weight,
                com_jitter_weight=v29_com_jitter_weight,
                foot_joint_indices=None,
                parents=parents,
                warmup_epochs=self.v29_physical_loss_warmup_epochs,
            )
        else:
            self.physical_space_temporal_loss_v29 = None

        # Optional v31 physical collision penalty (training only).
        self.use_physical_collision_penalty_v31 = use_physical_collision_penalty_v31
        self.v31_collision_loss_weight = v31_collision_loss_weight
        if self.use_physical_collision_penalty_v31:
            parents = None
            if self.j == 17:
                parents = H36M_17_PARENTS
            elif self.j == 28:
                parents = MPI_INF_3DHP_28_PARENTS
            self.physical_collision_penalty_v31 = PhysicalCollisionPenaltyV31(
                parents=parents,
                loss_weight=v31_collision_loss_weight,
                margin=v31_collision_margin,
                warmup_epochs=v31_collision_warmup_epochs,
            )
        else:
            self.physical_collision_penalty_v31 = None

        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Notify the model of the current epoch for loss warmups / curricula."""
        self.epoch = epoch
        if (
            self.use_physical_space_temporal_loss_v29
            and self.physical_space_temporal_loss_v29 is not None
            and hasattr(self.physical_space_temporal_loss_v29, "set_epoch")
        ):
            self.physical_space_temporal_loss_v29.set_epoch(epoch)
        if (
            self.use_physical_collision_penalty_v31
            and self.physical_collision_penalty_v31 is not None
            and hasattr(self.physical_collision_penalty_v31, "set_epoch")
        ):
            self.physical_collision_penalty_v31.set_epoch(epoch)

        # Make sure the ST transformer can accept an additive attention mask even
        # when epipolar bias is disabled.
        if not self.use_epipolar_bias:
            self.st_transformer = nn.ModuleList(
                [
                    EpipolarBiasedTransformerEncoderLayer(
                        d_model=self.d,
                        nhead=self.n_heads,
                        dim_feedforward=self.d * 4,
                        dropout=0.1,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(len(self.st_transformer))
                ]
            )
            self.epipolar_bias = None

    def _prepare_view_mask(
        self,
        view_mask: Optional[torch.Tensor],
        B: int,
        T: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize ``view_mask`` to ``(B * T, V)``.

        Accepted shapes: ``(B, T, V)``, ``(B, V)``, ``(N, V)`` where ``N`` is
        already ``B * T``.
        """
        if view_mask is None:
            return torch.ones(B * T, V, device=device)

        if view_mask.dim() == 2:
            # (N, V) or (B, V)
            if view_mask.shape[0] == B and view_mask.shape[1] == V:
                return view_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, V)
            elif view_mask.shape[0] == B * T and view_mask.shape[1] == V:
                return view_mask
            else:
                raise ValueError(
                    f"view_mask (N, V) shape {view_mask.shape} incompatible with "
                    f"B={B}, T={T}, V={V}"
                )
        elif view_mask.dim() == 3:
            # (B, T, V)
            if view_mask.shape != (B, T, V):
                raise ValueError(
                    f"view_mask (B, T, V) shape {view_mask.shape} incompatible with "
                    f"B={B}, T={T}, V={V}"
                )
            return view_mask.reshape(B * T, V)
        else:
            raise ValueError(
                f"view_mask must have shape (B, T, V), (B, V) or (N, V), got {view_mask.shape}"
            )

    def _build_view_attention_mask(
        self,
        view_mask_flat: torch.Tensor,
        B: int,
        T: int,
        V: int,
        n_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive attention mask for the time+view transformer.

        Tokens from masked-out views are blocked from attending to any token and
        from being attended to by any token.  The returned tensor is ``0`` for
        allowed attention and ``-1e9`` for blocked attention, matching the
        convention of ``nn.MultiheadAttention`` additive masks.

        Args
        ----
        view_mask_flat:
            ``(B * T, V)`` binary mask.

        Returns
        -------
        ``(B*J*n_heads, T*V, T*V)`` additive mask ready for ``attn_mask=``.
        """
        # mask_per_view: (B, T, V)
        mask_per_view = view_mask_flat.view(B, T, V)
        # token_mask: (B, T, V) -> (B, T*V)
        token_mask = mask_per_view.reshape(B, T * V)
        # Block masked tokens: attn_mask[b, i, j] is blocked if either token is masked.
        valid = token_mask.unsqueeze(2) * token_mask.unsqueeze(1)  # (B, T*V, T*V)
        valid = valid.unsqueeze(1).expand(-1, n_heads, -1, -1)  # (B, n_heads, T*V, T*V)
        # Expand to joint dimension: each joint uses the same temporal/view layout.
        valid = valid.unsqueeze(2).expand(-1, -1, self.j, -1, -1)  # (B, n_heads, J, T*V, T*V)
        valid = valid.permute(0, 2, 1, 3, 4).reshape(B * self.j * n_heads, T * V, T * V)
        valid = valid.float()
        return valid * 0.0 + (1.0 - valid) * -1e9

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[object] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        # Defensive: input data may contain NaN/Inf for occluded/missing joints.
        # Replace them with finite placeholders while keeping the confidence channel
        # (used downstream for masking) intact.
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_mask = torch.isnan(x) | torch.isinf(x)
            x = torch.where(nan_mask, torch.zeros_like(x), x)
        if K is not None and (torch.isnan(K).any() or torch.isinf(K).any()):
            K = torch.nan_to_num(K, nan=0.0, posinf=1e4, neginf=-1e4)
        if R is not None and (torch.isnan(R).any() or torch.isinf(R).any()):
            R = torch.nan_to_num(R, nan=0.0, posinf=1e4, neginf=-1e4)
        if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
            t = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
        if view_mask is not None and (torch.isnan(view_mask).any() or torch.isinf(view_mask).any()):
            view_mask = torch.nan_to_num(view_mask, nan=0.0, posinf=1.0, neginf=0.0)

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import (
                _cameras_to_tensors,
            )
            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        view_mask_flat = self._prepare_view_mask(view_mask, B, T, V, device)

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Apply view mask to confidences.
        confidences = confidences * view_mask_flat.unsqueeze(-1)

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Optional rotation correction on extrinsics.
        if self.use_rotation_correction and self.rotation_correction_head is not None:
            feat_rot = self._extract_frame_features(x_flat, K_corrected, R, t)
            feat_rot_pooled = feat_rot.mean(dim=2)  # (B*T, V, d)
            R, _ = self.rotation_correction_head(feat_rot_pooled, R)

        # Per-frame v3 features (uses corrected intrinsics and possibly corrected R).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)

        # Optional dense joint-level self-attention (per-view).
        if self.omni_joint_attn is not None:
            feat_j = feat.permute(0, 2, 1, 3).reshape(B * T * V, J, self.d)
            for layer in self.omni_joint_attn:
                feat_j = layer(feat_j)
            feat = feat_j.view(B * T, V, J, self.d)

        # Graph-joint attention over (view, joint) skeleton graph.
        feat = self._apply_graph_joint_attention(feat, J)

        # Camera conditioning.
        if self.camera_conditioning is not None:
            feat = self.camera_conditioning(feat, K_corrected, R, t)

        # Hierarchical multi-scale temporal/cross-view fusion.
        if self.multiscale_fusion is not None:
            feat = feat.view(B, T, V, J, self.d)
            feat = self.multiscale_fusion(feat)
            feat = feat.view(B * T, V, J, self.d)

        # View embedding: learned positional embedding plus optional camera
        # conditioned embedding as a residual.  Keeping the learned embedding helps
        # fixed-view accuracy while the camera embedding enables variable views.
        feat = feat.view(B, T, V, J, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + view_emb
        if self.use_camera_view_embedding and self.camera_view_embedding is not None:
            camera_emb = self.camera_view_embedding(K_corrected, R, t)  # (B*T, V, d)
            camera_emb = camera_emb.view(B, T, V, 1, self.d)
            feat = feat + camera_emb

        # Optional permutation-invariant view aggregator over views.
        if self.use_perceiver_aggregator and self.perceiver_aggregator is not None:
            feat = self.perceiver_aggregator(feat, view_mask=view_mask)
        elif self.use_set_view_aggregator and self.set_view_aggregator is not None:
            feat = self.set_view_aggregator(feat, view_mask=view_mask)

        # Optional cross-view transformer (v17) with geometric ray/camera embeddings.
        if self.use_cross_view_transformer_v17 and self.cross_view_transformer_v17 is not None:
            # v17 expects 5D points_2d (B, T, V, J, 2); reshape from the internal 4D form.
            points_2d_5d = points_2d.view(B, T, V, J, 2)
            feat = self.cross_view_transformer_v17(
                feat,
                K=K_corrected,
                R=R,
                t=t,
                points_2d=points_2d_5d,
                view_mask=view_mask,
            )

        # Optional deformable cross-view attention (v18) guided by epipolar geometry.
        if self.use_deformable_cross_view_attention_v18 and self.deformable_cross_view_attention_v18 is not None:
            feat = self.deformable_cross_view_attention_v18(
                feat,
                K=K_corrected,
                R=R,
                t=t,
                points_2d=points_2d,
                view_mask=view_mask,
            )

        # Optional v29/v30 hierarchical multi-scale view encoder.
        if self.use_hierarchical_multiview_v29 and self.hierarchical_multiview_v29 is not None:
            feat = feat + self.hierarchical_multiview_v29(feat, view_mask=view_mask_flat.view(B, T, V))
        if self.use_hierarchical_multiview_v30 and self.hierarchical_multiview_v30 is not None:
            feat = feat + self.hierarchical_multiview_v30(feat, view_mask=view_mask_flat.view(B, T, V))
        if self.use_hierarchical_multiview_v31 and self.hierarchical_multiview_v31 is not None:
            feat = feat + self.hierarchical_multiview_v31(
                feat,
                view_mask=view_mask_flat.view(B, T, V),
                points_2d=points_2d.view(B, T, V, J, 2),
                K=K_corrected.view(B, T, V, 3, 3),
                R=R.view(B, T, V, 3, 3),
                t=t.view(B, T, V, 3),
            )

        # Optional domain embedding (useful when mixing H36M/MPI/WebBridge).
        if self.use_domain_embedding and domain_id is not None:
            domain_emb = self.domain_embedding(domain_id)  # (B, d)
            feat = feat + domain_emb.view(B, 1, 1, 1, self.d)

        # Spatio-temporal (time + view) attention with optional epipolar bias.
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        feat = feat + time_emb
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)

        # Prepare an additive view mask for the ST transformer.
        st_view_mask = self._build_view_attention_mask(
            view_mask_flat, B, T, V, self.n_heads, device
        )

        if self.use_epipolar_bias and self.epipolar_bias is not None:
            from motionflow_mv.fusion.epipolar_transformer_bias import (
                build_temporal_bias_from_frames,
            )

            epi_bias = self.epipolar_bias(K_corrected, R, t, points_2d)
            epi_bias = epi_bias.view(B, T, V, V)
            # Combine epipolar bias with the view mask via addition (both are
            # additive masks, with -1e9 for blocked positions).
            attn_mask = build_temporal_bias_from_frames(
                epi_bias, n_heads=self.n_heads, n_joints=J
            )
            attn_mask = attn_mask + st_view_mask
            for layer in self.st_transformer:
                feat = layer(feat, epipolar_bias=attn_mask)
        else:
            # The ST transformer has been replaced with EpipolarBiasedTransformerEncoder
            # layers, so they accept an additive attn_mask even without epipolar bias.
            for layer in self.st_transformer:
                feat = layer(feat, epipolar_bias=st_view_mask)

        feat = feat.view(B, J, T, V, self.d)
        # Zero out tokens from masked views so NaNs produced by softmax over
        # fully-blocked positions cannot leak downstream in variable-view mode.
        st_mask = view_mask_flat.view(B, T, V, 1, 1).permute(0, 3, 1, 2, 4)
        feat = torch.where(st_mask.bool(), feat, torch.zeros_like(feat))
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Anisotropic covariance prediction per (view, joint).
        raw_cov = self.covariance_head(feat)
        L = self._cholesky_to_covariance(raw_cov)
        precision = 1.0 / (
            L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
        )

        # Visibility gating: optional context-aware head or v3 fallback.
        visibility = self._visibility_multiplier(feat, confidences)
        # Defensive: visibility head can produce NaN/Inf when upstream features
        # are corrupted; clamp to a valid probability before returning / using
        # it in BCE losses downstream.
        visibility = torch.nan_to_num(visibility, nan=0.5, posinf=1.0, neginf=0.0)
        visibility = visibility.clamp(0.0, 1.0)

        # Per-frame weight prediction and triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)

        # Defensive: weight head can produce NaN when upstream features are NaN/Inf
        # (e.g. from corrupted input data or degenerate camera geometry).
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.nan_to_num(weights, nan=1e-4, posinf=1e4, neginf=1e-4)

        # Optional adaptive view selection.
        budget_loss = torch.tensor(0.0, device=device)
        if self.use_adaptive_view_selection and self.adaptive_view_selector is not None:
            selector_mask, budget_loss = self.adaptive_view_selector(feat)
            weights = weights * selector_mask

        # Apply view mask to weights before triangulation.
        weights = weights * view_mask_flat.unsqueeze(-1)
        if self.use_full_precision_dlt:
            weights = weights * confidences * visibility
            weights = weights.clamp(min=1e-4, max=1e4)
            # Regularise covariance to avoid singular precision matrices.
            eye2 = torch.eye(2, device=L.device, dtype=L.dtype).view(1, 1, 1, 2, 2)
            cov = L @ L.transpose(-2, -1) + 1e-3 * eye2
            # Extra robustness: if cov is still singular/near-singular, add a
            # larger ridge before inversion.
            try:
                precision_matrix = torch.linalg.inv(cov)
            except RuntimeError:
                precision_matrix = torch.linalg.inv(cov + 1e-2 * eye2)
            if torch.isnan(precision_matrix).any() or torch.isinf(precision_matrix).any():
                precision_matrix = torch.where(
                    torch.isnan(precision_matrix) | torch.isinf(precision_matrix),
                    eye2.expand_as(precision_matrix),
                    precision_matrix,
                )
            # Clamp precision matrix to prevent degenerate / exploding Mahalanobis
            # distances in the robust reweight path.
            precision_matrix = precision_matrix.clamp(min=-1e3, max=1e3)
            Rt = torch.cat([R, t[..., None]], dim=-1)
            P = K_corrected @ Rt
            from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq
            pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights, precision_matrix=precision_matrix)
            if self.use_robust_dlt_reweight:
                # One-step robust reweighting based on predicted covariance.
                pred_3d_h = torch.cat(
                    [pred_3d_raw, torch.ones(pred_3d_raw.shape[0], pred_3d_raw.shape[1], 1, device=pred_3d_raw.device, dtype=pred_3d_raw.dtype)],
                    dim=-1,
                )  # (N, J, 4)
                x_h = (P.unsqueeze(2) @ pred_3d_h.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # (N, V, J, 3)
                x_pred = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)  # (N, V, J, 2)
                residual = x_pred - points_2d  # (N, V, J, 2)
                residual_col = residual.unsqueeze(-1)  # (N, V, J, 2, 1)
                mahal = residual_col.transpose(-2, -1) @ precision_matrix @ residual_col  # (N, V, J, 1, 1)
                mahal = mahal.squeeze(-1).squeeze(-1).clamp(min=0.0, max=50.0)
                rho = torch.exp(-mahal / 2.0).clamp(min=1e-3, max=1.0)
                # Detach robust weights so the second solve does not backprop through
                # the (potentially unstable) precision/covariance head.
                weights_robust = (weights * rho * view_mask_flat.unsqueeze(-1)).detach()
                weights_robust = weights_robust.clamp(min=1e-4, max=1e4)
                pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights_robust, precision_matrix=precision_matrix.detach())

                # Optional IRLS refinement of the robust weights using a Cauchy kernel
                # and MAD auto-scaling.  This is kept separate from the one-step
                # reweight above so existing checkpoints are unaffected.
                if self.use_irls_reweight:
                    for _ in range(self.irls_n_iters):
                        pred_3d_h = torch.cat(
                            [pred_3d_raw, torch.ones(pred_3d_raw.shape[0], pred_3d_raw.shape[1], 1, device=pred_3d_raw.device, dtype=pred_3d_raw.dtype)],
                            dim=-1,
                        )
                        x_h = (P.unsqueeze(2) @ pred_3d_h.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
                        x_pred = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
                        residual = x_pred - points_2d  # (N, V, J, 2)
                        # Robust scale via MAD over all residuals in the batch.
                        abs_res = residual.abs()  # (N, V, J, 2)
                        median = torch.median(abs_res)
                        mad = torch.median((abs_res - median).abs()) * 1.4826
                        scale = (mad + 1e-6).clamp(min=1e-3)
                        u = abs_res / scale
                        # Cauchy weight per coordinate, then average over (x, y).
                        w = 1.0 / (1.0 + (u / self.irls_cauchy_scale) ** 2)
                        w = w.mean(dim=-1).clamp(min=1e-4, max=1.0)  # (N, V, J)
                        weights_irls = (weights * w * view_mask_flat.unsqueeze(-1)).detach().clamp(min=1e-4, max=1e4)
                        pred_3d_raw = triangulate_dlt_batched_lstsq(
                            points_2d, P, weights_irls, precision_matrix=precision_matrix.detach()
                        )
        else:
            weights = weights * confidences * precision * visibility
            weights = weights.clamp(min=1e-4, max=1e4)
            Rt = torch.cat([R, t[..., None]], dim=-1)
            P = K_corrected @ Rt
            from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq
            pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        # Adaptive Gauss-Newton refinement.
        feat_pooled = feat.mean(dim=1)
        damping = self.damping_head(feat_pooled).squeeze(-1)
        damping = self.min_gn_damping + (
            self.max_gn_damping - self.min_gn_damping
        ) * damping

        from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
            _adaptive_gauss_newton,
        )

        pred_3d_gn = _adaptive_gauss_newton(
            points_2d,
            weights,
            K_corrected,
            R,
            t,
            pred_3d_raw,
            damping,
            num_iters=self.gn_iters,
        )

        # Optional v21 neural bundle-adjustment refinement of pose and cameras.
        if (
            self.use_neural_bundle_adjustment_v21
            and self.neural_bundle_adjustment_v21 is not None
        ):
            pred_3d_gn, K_corrected, R, t = self.neural_bundle_adjustment_v21(
                pred_3d_gn, points_2d, K_corrected, R, t, weights
            )

        # Optional v25 multi-view geometry fusion refinement.
        geom_loss_v25 = torch.tensor(0.0, device=device, dtype=pred_3d_gn.dtype)
        if (
            self.use_multiview_geometry_fusion_v25
            and self.multiview_geometry_fusion_v25 is not None
        ):
            pred_3d_gn_v25, geom_loss_v25 = self.multiview_geometry_fusion_v25(
                points_2d=points_2d.view(B, T, V, J, 2),
                K=K_corrected.view(B, T, V, 3, 3),
                R=R.view(B, T, V, 3, 3),
                t=t.view(B, T, V, 3),
                pred_3d_init=pred_3d_gn.view(B, T, J, 3),
                view_mask=view_mask_flat.view(B, T, V),
                confidence=confidences.view(B, T, V, J),
            )
            pred_3d_gn = pred_3d_gn_v25.view(B * T, J, 3)

        # Optional v27 test-time self-evolution.  Only active at eval to keep the
        # training graph simple and avoid extra forward passes.
        if (
            not self.training
            and self.use_test_time_self_evolution_v27
            and self.test_time_self_evolution_v27 is not None
        ):
            with torch.no_grad():
                pred_3d_gn_tte = self.test_time_self_evolution_v27(
                    pred_3d_gn.view(B, T, J, 3),
                    points_2d.view(B, T, V, J, 2),
                    K_corrected.view(B, T, V, 3, 3),
                    R.view(B, T, V, 3, 3),
                    t.view(B, T, V, 3),
                    view_mask=view_mask_flat.view(B, T, V),
                    confidence=confidences.view(B, T, V, J),
                )
                pred_3d_gn = pred_3d_gn_tte.view(B * T, J, 3)

        # Optional v29 test-time self-evolution (physical-space aware).
        if (
            not self.training
            and self.use_test_time_self_evolution_v29
            and self.test_time_self_evolution_v29 is not None
        ):
            with torch.no_grad():
                pred_3d_gn_tte29 = self.test_time_self_evolution_v29(
                    pred_3d_gn.view(B, T, J, 3),
                    points_2d.view(B, T, V, J, 2),
                    K_corrected.view(B, T, V, 3, 3),
                    R.view(B, T, V, 3, 3),
                    t.view(B, T, V, 3),
                    view_mask=view_mask_flat.view(B, T, V),
                    confidence=confidences.view(B, T, V, J),
                )
                pred_3d_gn = pred_3d_gn_tte29.view(B * T, J, 3)

        # Residual refinement head (deterministic MLP or diffusion-based v20).
        if self.use_diffusion_refiner_v20 and self.diffusion_refiner_v20 is not None:
            if self.training:
                t_diff = torch.randint(
                    0,
                    self.num_diffusion_steps,
                    (pred_3d_gn.shape[0],),
                    device=device,
                )
                pred_3d = self.diffusion_refiner_v20(
                    pred_3d_gn, feat=feat_pooled, t=t_diff
                )
            else:
                pred_3d = self.diffusion_refiner_v20(pred_3d_gn, feat=feat_pooled)
        else:
            residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d_gn + delta

        # Optional v32 temporal trajectory-consistency refiner.
        self._v32_loss = None
        if (
            self.use_trajectory_consistency_v32
            and self.trajectory_consistency_refiner is not None
            and T > 2
        ):
            pred_3d_raw_seq = pred_3d.view(B, T, self.j, 3)
            pred_3d_ref_seq = self.trajectory_consistency_refiner(pred_3d_raw_seq)
            pred_3d = pred_3d_ref_seq.view(B * T, self.j, 3)
            from motionflow_mv.fusion.trajectory_consistency_v32 import (
                trajectory_consistency_loss,
            )

            v32_smooth, v32_drift = trajectory_consistency_loss(
                pred_3d_ref_seq, pred_3d_raw_seq
            )
            self._v32_loss = (
                self.v32_smooth_weight * v32_smooth
                + self.v32_drift_weight * v32_drift
            )

        # Optional final kinematic-chain refiner.
        if self.use_kinematic_refiner and self.kinematic_refiner is not None:
            pred_3d = pred_3d + self.kinematic_refiner(pred_3d)

        # Epipolar consistency loss.
        epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
        epi_loss = self.epipolar_loss_weight * epi_loss + self.v25_geom_loss_weight * geom_loss_v25
        if self._v32_loss is not None:
            epi_loss = epi_loss + self._v32_loss

        # Optional entropy regularisation on triangulation weights.
        if (
            self.use_entropy_regularization
            and self.attention_entropy_loss is not None
        ):
            epi_loss = epi_loss + self.attention_entropy_loss(weights)

        pred_3d = pred_3d.view(B, T, J, 3)

        # Optional v22 kinematic anthropometric prior.
        if (
            self.use_kinematic_anthropometric_prior_v22
            and self.kinematic_anthropometric_prior_v22 is not None
        ):
            pred_3d_flat = pred_3d.view(B * T, J, 3)
            feat_pooled_flat = feat_pooled.view(B * T, J, self.d)
            pred_3d_refined, kap_loss = self.kinematic_anthropometric_prior_v22(
                feat_pooled_flat, pred_3d_flat
            )
            pred_3d = pred_3d_refined.view(B, T, J, 3)
            epi_loss = epi_loss + self.kap_loss_weight * kap_loss

        # Optional v19 temporal Perceiver refinement on the final per-frame 3D poses.
        if self.use_temporal_perceiver_v19 and self.temporal_perceiver_refiner_v19 is not None:
            # Feature-aware temporal perceiver: concat 3D pose with view-pooled ST features.
            feat_pooled_v19 = feat_pooled.view(B, T, J, self.d)
            temporal_input = torch.cat([pred_3d, feat_pooled_v19], dim=-1)
            pred_3d = self.temporal_perceiver_refiner_v19(temporal_input, baseline_3d=pred_3d)

        # Optional v28 physical-space alignment.
        if self.use_physical_space_alignment_v28 and self.physical_space_alignment_v28 is not None:
            pred_3d = self.physical_space_alignment_v28(pred_3d)
            if self.v28_floor_loss_weight > 0.0 or self.v28_bone_temporal_weight > 0.0:
                # Select the parent list and foot indices based on the skeleton.
                if self.j == 17:
                    parents = H36M_17_PARENTS
                elif self.j == 28:
                    parents = MPI_INF_3DHP_28_PARENTS
                else:
                    parents = list(range(-1, self.j - 1)) + [-1]

                if self.v28_floor_loss_weight > 0.0:
                    # Leaf joints are treated as feet/ankles for the floor loss.
                    children = [[] for _ in range(self.j)]
                    for child, parent in enumerate(parents):
                        if parent >= 0:
                            children[parent].append(child)
                    foot_indices = [j for j, c in enumerate(children) if len(c) == 0]
                    if len(foot_indices) == 0:
                        foot_indices = list(range(self.j))
                    floor_h = pred_3d[..., 1].min().detach()
                    v28_floor = floor_loss(pred_3d, floor_h, foot_indices)
                    epi_loss = epi_loss + self.v28_floor_loss_weight * v28_floor

                if self.v28_bone_temporal_weight > 0.0 and pred_3d.shape[1] > 1:
                    v28_bone = bone_temporal_loss(pred_3d, parents)
                    epi_loss = epi_loss + self.v28_bone_temporal_weight * v28_bone

        # Optional v29 physical-space temporal loss (training only).
        if (
            self.training
            and self.use_physical_space_temporal_loss_v29
            and self.physical_space_temporal_loss_v29 is not None
        ):
            v29_physical_loss, v29_physical_terms = self.physical_space_temporal_loss_v29(pred_3d)
            epi_loss = epi_loss + v29_physical_loss

        # Optional v31 physical collision penalty (training only).
        if (
            self.training
            and self.use_physical_collision_penalty_v31
            and self.physical_collision_penalty_v31 is not None
        ):
            collision_loss, _ = self.physical_collision_penalty_v31(pred_3d)
            epi_loss = epi_loss + collision_loss

        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)
        visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)
            visibility = visibility.squeeze(1)

        out = (pred_3d, weights, visibility, L, epi_loss)

        if self.return_pp_delta:
            out += (pp_delta,)
            if self.correct_focal:
                out += (focal_scale,)

        return out


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for smoke tests)."""
    from motionflow_mv.calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


if __name__ == "__main__":
    # T01 CPU smoke test: B=2, T=9, V=4, J=17.
    B, T, V, J = 2, 9, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    # Default v5 configuration (camera embedding + set aggregator off, v4 path).
    model = OmniMultiViewFusionV5(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    pred, weights, visibility, covariance, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert covariance.shape == (B, T, V, J, 2, 2)
    assert epi_loss.numel() == 1

    loss = pred.mean() + epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("OmniMultiViewFusionV5 default-toggle CPU smoke test passed (T=9)")

    # v5 with camera embedding + set aggregator enabled.
    model_full = OmniMultiViewFusionV5(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
        use_camera_view_embedding=True,
        use_set_view_aggregator=True,
        set_view_n_isab_layers=2,
        set_view_num_inducing_points=32,
    )
    pred2, weights2, visibility2, covariance2, epi_loss2 = model_full(
        x, cameras=cameras
    )
    assert pred2.shape == (B, T, J, 3)
    assert weights2.shape == (B, T, V, J)
    assert visibility2.shape == (B, T, V, J)
    assert covariance2.shape == (B, T, V, J, 2, 2)
    assert epi_loss2.numel() == 1
    loss2 = pred2.mean() + epi_loss2
    loss2.backward()
    assert any(p.grad is not None for p in model_full.parameters())
    print(
        "OmniMultiViewFusionV5 camera-embedding + set-aggregator CPU smoke test passed (T=9)"
    )

    # Variable view mask with V=2.
    V2 = 2
    cameras_v2 = _make_cameras(V2)
    x_v2 = torch.rand(B, T, V2, J, 3)
    view_mask = torch.zeros(B, T, V2)
    view_mask[:, :, 0] = 1.0
    view_mask[:, :, 1] = 0.0
    model_v2 = OmniMultiViewFusionV5(
        j=J,
        d=64,
        n_views=V2,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
        use_camera_view_embedding=True,
        use_set_view_aggregator=True,
    )
    pred3, weights3, visibility3, covariance3, epi_loss3 = model_v2(
        x_v2, cameras=cameras_v2, view_mask=view_mask
    )
    assert pred3.shape == (B, T, J, 3)
    assert weights3.shape == (B, T, V2, J)
    assert visibility3.shape == (B, T, V2, J)
    assert covariance3.shape == (B, T, V2, J, 2, 2)
    assert epi_loss3.numel() == 1
    # Masked-out view should have near-zero weights.
    assert weights3[:, :, 1, :].abs().max().item() < 1e-3
    print("OmniMultiViewFusionV5 variable-view mask CPU smoke test passed (V=2)")
