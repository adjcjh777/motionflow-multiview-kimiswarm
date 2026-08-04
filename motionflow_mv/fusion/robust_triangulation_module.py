"""FusionModule wrapper around RobustTriangulationModel.

This predicts per-view weights with a small transformer and solves a
differentiable weighted DLT system for each joint, using the calibrated
camera projection matrices.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .robust_triangulation import RobustTriangulationModel


class RobustTriangulationFusion(FusionModule):
    """Trainable per-view weighting + differentiable DLT as a plugin."""

    name = "robust_triangulation"

    def __init__(self, model: RobustTriangulationModel | None = None, j: int = 17, d: int = 32, n_views: int = 4):
        super().__init__()
        self.model = model or RobustTriangulationModel(j=j, d=d, n_views=n_views)
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

        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)
        proj_tensor = torch.from_numpy(proj_matrices).to(next(self.model.parameters()).device)

        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred = self.model(x_tensor, proj_tensor).cpu().numpy()
        return pred


def register_robust_triangulation_fusion_module() -> None:
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(RobustTriangulationFusion())
