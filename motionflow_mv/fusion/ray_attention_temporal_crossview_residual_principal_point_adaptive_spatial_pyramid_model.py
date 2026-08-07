"""Adaptive-scale cross-view spatial pyramid model.

Extends the principal-point model
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` by inserting
an ``AdaptiveCrossViewSpatialPyramid`` module after the per-frame encoder.  The
adaptive pyramid replaces the fixed multi-scale concatenation with a learned
soft scale-selection gate, allowing the model to adjust its effective joint-scale
receptive field per sample.
"""

import torch.nn as nn

from .adaptive_cross_view_spatial_pyramid import AdaptiveCrossViewSpatialPyramid
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveSpatialPyramid(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP model with adaptive spatial pyramid.

    Parameters
    ----------
    spatial_pyramid_scales:
        Downsample factors for the joint axis.  Default ``(1, 2, 4)``.
    spatial_pyramid_heads:
        Number of attention heads in each cross-view pyramid branch.  Default 1.
    gate_hidden:
        Hidden dimension of the scale-gating MLP.  Defaults to ``d // 2``.
    See the parent class for the remaining args.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        spatial_pyramid_scales: tuple = (1, 2, 4),
        spatial_pyramid_heads: int = 1,
        gate_hidden: int | None = None,
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
            return_visibility=return_visibility,
        )
        self.spatial_pyramid_scales = spatial_pyramid_scales
        self.spatial_pyramid = AdaptiveCrossViewSpatialPyramid(
            d=d,
            n_views=n_views,
            scales=spatial_pyramid_scales,
            n_heads=spatial_pyramid_heads,
            gate_hidden=gate_hidden,
        )

    def _extract_frame_features(self, x, K, R, t):
        """Run the base per-frame encoder and then the adaptive multi-scale pyramid."""
        feat = super()._extract_frame_features(x, K, R, t)  # (N, V, J, d)
        feat = self.spatial_pyramid(feat)
        return feat
