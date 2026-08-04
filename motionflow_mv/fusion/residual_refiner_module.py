"""FusionModule wrapper around ResidualRefinerModel.

This first triangulates a coarse baseline with DLT, then feeds it together
with the per-view 2D observations to ``ResidualRefinerModel`` for a small
residual correction.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .residual_refiner import ResidualRefinerModel
from .triangulation import triangulate_confidence_weighted


class ResidualRefinerFusion(FusionModule):
    """DLT baseline + learned residual correction as a plugin."""

    name = "residual_refiner"

    def __init__(self, model: ResidualRefinerModel | None = None, j: int = 17, d: int = 64, n_views: int = 5):
        super().__init__()
        self.model = model or ResidualRefinerModel(j=j, d=d, n_views=n_views)
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
        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

        # Compute DLT baseline per frame.
        baseline = np.zeros((T, J, 3), dtype=np.float64)
        for t in range(T):
            for j_idx in range(J):
                baseline[t, j_idx] = triangulate_confidence_weighted(
                    points_2d[t, :, j_idx, :],
                    proj_matrices,
                    confidences[t, :, j_idx],
                )
        baseline = baseline.astype(np.float32)

        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)
        baseline_tensor = torch.from_numpy(baseline).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred = self.model(x_tensor, baseline_tensor).cpu().numpy()
        return pred


def register_residual_refiner_fusion_module() -> None:
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(ResidualRefinerFusion())
