"""End-to-end multi-view pose fusion pipeline."""

from typing import List
import numpy as np

from .calibration.camera import Camera
from .fusion.triangulation import triangulate_confidence_weighted


class MultiViewPipeline:
    """Pipeline: N views -> per-view 2D keypoints + confidences -> fused 3D skeleton."""

    def __init__(self, estimator):
        """Args:
            estimator: object with `.extract(video_path)` returning dict with
                       'keypoints_2d' (T, J, 2) and 'confidence' (T, J).
        """
        self.estimator = estimator

    def fuse_frame(self, points_2d: np.ndarray, confidences: np.ndarray, cameras: List[Camera]) -> np.ndarray:
        """Fuse one frame of multi-view 2D keypoints into a 3D skeleton.

        Args:
            points_2d: (V, J, 2)
            confidences: (V, J)
            cameras: list of V Camera objects.

        Returns:
            (J, 3) world-coordinate 3D skeleton.
        """
        points_2d = np.asarray(points_2d)
        confidences = np.asarray(confidences)
        v, j, _ = points_2d.shape
        proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)

        joints_3d = np.zeros((j, 3), dtype=np.float64)
        for joint_idx in range(j):
            joints_3d[joint_idx] = triangulate_confidence_weighted(
                points_2d[:, joint_idx, :],
                proj_matrices,
                confidences[:, joint_idx],
            )
        return joints_3d
