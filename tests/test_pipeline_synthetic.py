"""Synthetic end-to-end test of the multi-view fusion pipeline."""

import numpy as np
import pytest

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe


def _make_cameras(n_views: int = 4, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(123)
    cameras = []
    for i in range(n_views):
        # Random intrinsics
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)
        # Random rotation (proper)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        # Place camera on a sphere looking toward origin
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _random_skeleton(j: int = 17, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5]) + np.array([0.0, 0.0, 3.0])


def test_dlt_pipeline_synthetic():
    """Project a known 3D skeleton into multiple views and recover it."""
    rng = np.random.default_rng(2024)
    j = 17
    X_world = _random_skeleton(j, rng)
    cameras = _make_cameras(4, rng)

    points_2d = []
    confidences = []
    for cam in cameras:
        P = cam.projection_matrix
        X_h = np.hstack([X_world, np.ones((j, 1))])  # (J, 4)
        x_h = (P @ X_h.T).T
        x = x_h[:, :2] / x_h[:, 2:3]
        points_2d.append(x)
        confidences.append(np.ones(j))

    points_2d = np.stack(points_2d, axis=0)  # (V, J, 2)
    confidences = np.stack(confidences, axis=0)  # (V, J)

    class _DummyEstimator:
        def extract(self, video_path: str):
            return {
                "keypoints_2d": points_2d[None],  # (1, V, J, 2)
                "confidence": confidences[None],
            }

    pipeline = MultiViewPipeline(estimator=_DummyEstimator())
    pred_3d = pipeline.fuse_frame(points_2d, confidences, cameras)

    error = mpjpe(pred_3d, X_world)
    pa_error = pa_mpjpe(pred_3d, X_world)
    print(f"Synthetic MPJPE: {error:.4f} mm, PA-MPJPE: {pa_error:.4f} mm")

    assert error < 1e-3, f"DLT pipeline failed to recover skeleton, MPJPE={error}"
    assert pa_error < 1e-3, f"DLT pipeline failed after alignment, PA-MPJPE={pa_error}"


if __name__ == "__main__":
    test_dlt_pipeline_synthetic()
    print("synthetic pipeline test passed")
