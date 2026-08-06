"""FusionModule wrapper around the spatial-pyramid anchor model.

Makes ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid``
available to ``MultiViewFusionPlugin`` and the rest of the MotionFlow pipeline.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid,
)


class RayAttentionTemporalCrossviewResidualPrincipalPointSpatialPyramidFusionModule(FusionModule):
    """Cross-view temporal ray-attention fusion with multi-scale spatial pyramid.

    Parameters
    ----------
    j:
        Number of body joints (28 for MPI-INF-3DHP, 17 for H36M).
    d:
        Feature dimension. Match the checkpoint.
    n_views:
        Maximum number of camera views. Match the checkpoint.
    n_st_layers:
        Number of cross-view spatio-temporal transformer layers.
    residual_hidden:
        Hidden dimension of the residual refinement MLP.
    checkpoint_path:
        Optional path to a trained checkpoint.
    input_scale:
        Scale factor converting input camera units to meters.
    spatial_pyramid_scales:
        Downsample factors for the multi-scale cross-view spatial pyramid.
    """

    name = "ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid"

    def __init__(
        self,
        j: int = 28,
        d: int = 64,
        n_views: int = 14,
        n_st_layers: int = 2,
        residual_hidden: int = 128,
        checkpoint_path: str | None = None,
        input_scale: float = 1.0,
        spatial_pyramid_scales: tuple = (1, 2, 4),
    ):
        super().__init__()
        self.input_scale = input_scale
        self.model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSpatialPyramid(
            j=j,
            d=d,
            n_views=n_views,
            n_st_layers=n_st_layers,
            residual_hidden=residual_hidden,
            spatial_pyramid_scales=spatial_pyramid_scales,
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys when loading checkpoint: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected keys when loading checkpoint: {unexpected[:5]}")
        self.model.eval()

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        points_2d = np.asarray(points_2d, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        if points_2d.ndim == 3:
            points_2d = points_2d[None]
        if confidences.ndim == 2:
            confidences = confidences[None]

        if self.input_scale != 1.0:
            cameras = [
                Camera(
                    K=cam.K.copy(),
                    R=cam.R.copy(),
                    t=cam.t.copy() / self.input_scale,
                )
                for cam in cameras
            ]

        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, _ = self.model(x_tensor, cameras=cameras)
            pred = pred.cpu().numpy()
        return pred


def register_ray_attention_temporal_crossview_residual_principal_point_spatial_pyramid_fusion_module() -> None:
    """Register the cross-view residual + PP + spatial-pyramid fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(RayAttentionTemporalCrossviewResidualPrincipalPointSpatialPyramidFusionModule())
