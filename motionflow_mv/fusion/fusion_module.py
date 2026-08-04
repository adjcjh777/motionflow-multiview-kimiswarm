"""Plugin interface for multi-view fusion modules.

All fusion backends (DLT, learned attention, temporal, etc.) implement the
same ``FusionModule`` contract so that ``MultiViewAdapter`` can swap them
without changing the rest of the pipeline.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..calibration.camera import Camera
from .triangulation import triangulate_confidence_weighted


class FusionModule(ABC):
    """Abstract base class for multi-view fusion backends.

    Subclasses implement ``fuse`` which consumes per-view 2D observations and
    calibrated cameras, and returns a single 3D skeleton per frame.
    """

    name: str = "abstract"

    @abstractmethod
    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        """Fuse per-view 2D keypoints into 3D joints.

        Args:
            points_2d: (T, V, J, 2) array of 2D keypoints.
            confidences: (T, V, J) array of confidence scores.
            cameras: list of V Camera objects.

        Returns:
            joints_3d: (T, J, 3) array of world-coordinate 3D joints.
        """
        ...


class DLTFusion(FusionModule):
    """Confidence-weighted DLT triangulation (geometry baseline)."""

    name = "dlt"

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        points_2d = np.asarray(points_2d, dtype=np.float64)
        confidences = np.asarray(confidences, dtype=np.float64)
        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

        T, V, J, _ = points_2d.shape
        joints_3d = np.zeros((T, J, 3), dtype=np.float64)
        for t in range(T):
            for j in range(J):
                joints_3d[t, j] = triangulate_confidence_weighted(
                    points_2d[t, :, j, :],
                    proj_matrices,
                    confidences[t, :, j],
                )
        return joints_3d


class FusionModuleRegistry:
    """Registry of named fusion modules."""

    def __init__(self):
        self._modules: dict[str, FusionModule] = {}

    def register(self, module: FusionModule) -> None:
        self._modules[module.name] = module

    def get(self, name: str) -> FusionModule:
        if name not in self._modules:
            raise KeyError(f"Unknown fusion module: {name}. Available: {list(self._modules)}")
        return self._modules[name]

    def names(self) -> List[str]:
        return list(self._modules.keys())


# Global registry populated with the built-in geometric baseline.
FUSION_REGISTRY = FusionModuleRegistry()
FUSION_REGISTRY.register(DLTFusion())
