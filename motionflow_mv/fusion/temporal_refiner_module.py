"""FusionModule wrapper around TemporalRefinerModel.

This builds per-frame DLT baselines, then slides a temporal window over the
sequence and refines the center frame of each window with a bidirectional GRU.
Boundary frames (which cannot be centered) are returned as-is from DLT.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .temporal_refiner import TemporalRefinerModel
from .triangulation import triangulate_confidence_weighted


class TemporalRefinerFusion(FusionModule):
    """DLT baseline + temporal GRU refinement as a plugin."""

    name = "temporal_refiner"

    def __init__(self, model: TemporalRefinerModel | None = None, j: int = 17, d: int = 64, n_views: int = 5, window: int = 7):
        super().__init__()
        self.model = model or TemporalRefinerModel(j=j, d=d, n_views=n_views)
        self.window = window
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
        half = self.window // 2

        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

        # Compute DLT baseline per frame.
        baseline = np.zeros((T, J, 3), dtype=np.float32)
        for t in range(T):
            for j_idx in range(J):
                baseline[t, j_idx] = triangulate_confidence_weighted(
                    points_2d[t, :, j_idx, :],
                    proj_matrices,
                    confidences[t, :, j_idx],
                )

        # Boundary frames: use DLT directly.
        refined = baseline.copy()
        if T < self.window:
            return refined

        device = next(self.model.parameters()).device
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        baseline_tensor = torch.from_numpy(baseline).unsqueeze(0).to(device)

        with torch.no_grad():
            for t in range(half, T - half):
                x_win = x_tensor[:, t - half:t + half + 1, :, :, :]
                baseline_win = baseline_tensor[:, t - half:t + half + 1, :, :]
                refined[t] = self.model(x_win, baseline_win).cpu().numpy().squeeze(0)

        return refined


def register_temporal_refiner_fusion_module() -> None:
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(TemporalRefinerFusion())
