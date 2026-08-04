"""FusionModule wrapper around the ray-aware attention model."""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_model import RayAttentionFusionModel


class RayAttentionFusionModule(FusionModule):
    """Ray-aware attention fusion with differentiable DLT triangulation.

    Uses per-view 2D keypoints, confidences and calibrated cameras to compute
    camera rays, then predicts per-view weights and triangulates.
    """

    name = "ray_attention"

    def __init__(self, model: RayAttentionFusionModel | None = None, j: int = 17, d: int = 64, n_views: int = 4):
        super().__init__()
        self.model = model or RayAttentionFusionModel(j=j, d=d, n_views=n_views)
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

        T, V, J, _ = points_2d.shape
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, _ = self.model(x_tensor, cameras)  # (T, J, 3)
            pred = pred.cpu().numpy()
        return pred


def register_ray_attention_fusion_module() -> None:
    """Register the ray-aware attention fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(RayAttentionFusionModule())
