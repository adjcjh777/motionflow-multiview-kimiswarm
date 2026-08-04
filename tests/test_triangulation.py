"""Unit tests for DLT triangulation."""

import numpy as np
import pytest

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import triangulate_dlt


def _random_camera(rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(0)
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 800.0
    K[:2, 2] = rng.uniform(300, 340, size=2)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = rng.standard_normal(3) * 2.0
    return Camera(K=K, R=R, t=t)


def test_triangulate_dlt_known_point():
    """Project a known 3D point into 4 views and recover it."""
    X_world = np.array([1.0, 0.5, 3.0])
    rng = np.random.default_rng(42)
    cameras = [_random_camera(rng) for _ in range(4)]

    points_2d = []
    proj_matrices = []
    for cam in cameras:
        P = cam.projection_matrix
        x_h = P @ np.append(X_world, 1.0)
        x = x_h[:2] / x_h[2]
        points_2d.append(x)
        proj_matrices.append(P)

    X_recovered = triangulate_dlt(np.array(points_2d), np.array(proj_matrices))
    np.testing.assert_allclose(X_recovered, X_world, atol=1e-4)


if __name__ == "__main__":
    test_triangulate_dlt_known_point()
    print("triangulation test passed")
