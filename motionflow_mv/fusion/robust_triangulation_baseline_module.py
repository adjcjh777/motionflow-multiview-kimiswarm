"""FusionModule wrapper around the IRLS/Charbonnier robust triangulation baseline."""

from typing import List

import numpy as np
import torch

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .robust_triangulation_baseline import triangulate_irls


def _projection_matrices(cameras: List[Camera]) -> np.ndarray:
    """Compute (V, 3, 4) projection matrices using PyTorch to avoid numpy BLAS crashes."""
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0))
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0))
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0))
    Rt = torch.cat([R, t[..., None]], dim=-1)  # (V, 3, 4)
    P = K @ Rt  # (V, 3, 4)
    return P.numpy()


class RobustTriangulationBaselineFusion(FusionModule):
    """Parameter-free IRLS triangulation baseline for multi-view fusion.

    This plugin triangulates each joint independently.  It starts from a
    confidence-weighted DLT solution and reweights views according to the
    Charbonnier robust loss for a fixed number of iterations.  No neural
    network is used, so it is deterministic and fast on CPU.
    """

    name = "robust_triangulation_baseline"

    def __init__(self, n_iters: int = 5, eps: float = 2.0):
        super().__init__()
        self.n_iters = n_iters
        self.eps = eps

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        """Fuse per-view 2D keypoints into 3D joints using IRLS triangulation.

        Args:
            points_2d: (T, V, J, 2) array of 2D keypoints.
            confidences: (T, V, J) array of confidence scores.
            cameras: list of V Camera objects.

        Returns:
            joints_3d: (T, J, 3) array of world-coordinate 3D joints.
        """
        points_2d = np.asarray(points_2d, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)

        if points_2d.ndim == 3:
            points_2d = points_2d[None]
        if confidences.ndim == 2:
            confidences = confidences[None]

        proj_matrices = _projection_matrices(cameras)
        T, V, J, _ = points_2d.shape
        joints_3d = np.zeros((T, J, 3), dtype=np.float64)
        for t in range(T):
            for j in range(J):
                joints_3d[t, j] = triangulate_irls(
                    points_2d[t, :, j, :],
                    proj_matrices,
                    n_iters=self.n_iters,
                    eps=self.eps,
                    confidences=confidences[t, :, j],
                )
        return joints_3d


def register_robust_triangulation_baseline_fusion_module() -> None:
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(RobustTriangulationBaselineFusion())
