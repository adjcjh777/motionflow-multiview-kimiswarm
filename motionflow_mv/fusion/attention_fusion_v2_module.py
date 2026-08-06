"""FusionModule wrapper around AttentionFusionModelV2 (geometry-aware).

This plugin feeds flattened camera projection matrices alongside the 2D
keypoints and confidences, letting the attention model reason about geometry.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .attention_model_v2 import AttentionFusionModelV2
from .fusion_module import FusionModule, FUSION_REGISTRY


class AttentionFusionV2Module(FusionModule):
    """Geometry-aware attention fusion wrapped as a FusionModule plugin."""

    name = "attention_v2"

    def __init__(self, model: AttentionFusionModelV2 | None = None, j: int = 17, d: int = 64, n_views: int = 4):
        super().__init__()
        self.model = model or AttentionFusionModelV2(j=j, d=d, n_views=n_views)
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

        # Flatten projection matrices for each view.
        proj = np.stack([cam.projection_matrix for cam in cameras], axis=0)  # (V, 3, 4)
        proj_flat = proj.reshape(V, 12).astype(np.float32)
        proj_batch = np.tile(proj_flat[None, :, :], (T, 1, 1))  # (T, V, 12)

        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)
        proj_tensor = torch.from_numpy(proj_batch).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred = self.model(x_tensor, proj_tensor).cpu().numpy()  # (T, J, 3)
        return pred


def register_attention_v2_fusion_module() -> None:
    FUSION_REGISTRY.register(AttentionFusionV2Module())
