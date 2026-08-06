"""FusionModule wrapper around the adaptive temporal-window pyramid model.

Makes ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid``
available to ``MultiViewFusionPlugin`` and the MotionFlow training/evaluation
pipeline under the name
``ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid``.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid,
)


class RayAttentionTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramidFusionModule(FusionModule):
    """Cross-view temporal residual + PP + adaptive temporal-window pyramid.

    Parameters
    ----------
    j:
        Number of body joints (e.g. 28 for MPI-INF-3DHP, 17 for H36M).
    d:
        Feature dimension.  Match the checkpoint.
    n_views:
        Maximum number of camera views.  Match the checkpoint.
    n_st_layers:
        Number of cross-view spatio-temporal transformer layers (kept for API
        compatibility; the pyramid replaces them).
    residual_hidden:
        Hidden dimension of the residual refinement MLP.
    temporal_scales:
        Temporal window sizes for the pyramid.
    pyramid_layers:
        Number of stacked pyramid layers.
    checkpoint_path:
        Optional path to a trained checkpoint.
    input_scale:
        Scale factor converting input camera units to meters.
    """

    name = "ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid"

    def __init__(
        self,
        j: int = 28,
        d: int = 64,
        n_views: int = 14,
        n_st_layers: int = 2,
        residual_hidden: int = 128,
        temporal_scales: tuple = (3, 7, 0),
        pyramid_layers: int = 1,
        checkpoint_path: str | None = None,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid(
            j=j,
            d=d,
            n_views=n_views,
            n_st_layers=n_st_layers,
            residual_hidden=residual_hidden,
            temporal_scales=temporal_scales,
            pyramid_layers=pyramid_layers,
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


def register_ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_fusion_module() -> None:
    """Register the adaptive window pyramid fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(
        RayAttentionTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramidFusionModule()
    )
