"""FusionModule wrapper around the trainable AttentionFusionModel.

This bridges the plugin interface (numpy arrays, calibrated cameras) with the
existing PyTorch attention fusion model. Camera parameters are currently
ignored by ``AttentionFusionModel``; this wrapper flattens them into the
input for future extensions.
"""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .attention_model import AttentionFusionModel
from .fusion_module import FusionModule


class AttentionFusionModule(FusionModule):
    """Trainable attention fusion wrapped as a ``FusionModule`` plugin.

    The underlying ``AttentionFusionModel`` expects a tensor of shape
    (B, V, J, 3) where the last dimension is (x, y, confidence).
    """

    name = "attention"

    def __init__(self, model: AttentionFusionModel | None = None, j: int = 17, d: int = 32, n_views: int = 4):
        super().__init__()
        self.model = model or AttentionFusionModel(j=j, d=d, n_views=n_views)
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
            # Single frame (V, J, 2)
            points_2d = points_2d[None]
        if confidences.ndim == 2:
            confidences = confidences[None]

        T, V, J, _ = points_2d.shape
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()  # (T, J, 3)
        return pred


def register_attention_fusion_module() -> None:
    """Register the attention fusion module with the global registry."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(AttentionFusionModule())
