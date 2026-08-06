"""Anchor model + adaptive scale-gated cross-view spatial pyramid.

Extends the iter14 best model
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` by inserting an
``AdaptiveScaleCrossViewSpatialPyramid`` module after the per-frame encoder.  The
gated pyramid lets the model learn which spatial scale to trust for each joint,
rather than fusing all scales with a fixed MLP.  This can improve robustness on
occluded or noisy views where fine-scale detail is unreliable.
"""

import torch.nn as nn

from .cross_view_spatial_pyramid import AdaptiveScaleCrossViewSpatialPyramid
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveScalePyramid(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP model with an adaptive scale-gated pyramid.

    Parameters
    ----------
    spatial_pyramid_scales:
        Downsample factors for the joint axis used by the pyramid.  Default
        ``(1, 2, 4)``.
    spatial_pyramid_heads:
        Number of attention heads in each cross-view pyramid branch.  Default 1.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining args.
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
        return_raw: bool = False,
        spatial_pyramid_scales: tuple = (1, 2, 4),
        spatial_pyramid_heads: int = 1,
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
            return_raw=return_raw,
        )
        self.spatial_pyramid_scales = spatial_pyramid_scales
        self.spatial_pyramid = AdaptiveScaleCrossViewSpatialPyramid(
            d=d,
            n_views=n_views,
            scales=spatial_pyramid_scales,
            n_heads=spatial_pyramid_heads,
        )

    def _extract_frame_features(self, x, K, R, t):
        """Run the base per-frame encoder and then the adaptive multi-scale pyramid."""
        feat = super()._extract_frame_features(x, K, R, t)  # (N, V, J, d)
        feat = self.spatial_pyramid(feat)
        return feat
