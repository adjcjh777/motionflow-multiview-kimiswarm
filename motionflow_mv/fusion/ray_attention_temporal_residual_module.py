"""FusionModule wrapper around the temporal residual ray-attention model."""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


class RayAttentionTemporalResidualFusionModule(FusionModule):
    """Temporal ray-attention fusion with a residual refinement head.

    Wraps ``RayAttentionFusionModelTemporalResidual`` so it can be used as a
    drop-in ``FusionModule`` in the MotionFlow multi-view pipeline.  The model
    consumes a temporal clip of 2D keypoints and confidences together with the
    calibrated camera rig, and returns a 3D pose trajectory.

    The model expects all inputs in a consistent metric unit (default meters).
    Use ``input_scale`` to tell the plugin the unit of the input cameras;
    outputs are always in meters.
    """

    name = "ray_attention_temporal_residual"

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        checkpoint_path: str | None = None,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.model = RayAttentionFusionModelTemporalResidual(j=j, d=d, n_views=n_views)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
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

        # Normalise cameras to meters.
        if self.input_scale != 1.0:
            cameras = [
                Camera(
                    K=cam.K.copy(),
                    R=cam.R.copy(),
                    t=cam.t.copy() / self.input_scale,
                )
                for cam in cameras
            ]

        T, V, J, _ = points_2d.shape
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, _ = self.model(x_tensor, cameras)  # (T, J, 3)
            pred = pred.cpu().numpy()
        return pred


def register_ray_attention_temporal_residual_fusion_module() -> None:
    """Register the temporal residual ray-attention fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(RayAttentionTemporalResidualFusionModule())
