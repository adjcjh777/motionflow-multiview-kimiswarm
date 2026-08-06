"""Smoke tests for the MotionFlow MultiViewFusionPlugin pipeline integration."""

import numpy as np
import pytest

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.ir.human_motion_ir import HumanMotionIR
from motionflow_mv.pipeline_multiview_plugin import (
    MultiViewFusionPlugin,
    create_multiview_plugin,
)


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras."""
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _project_skeleton(X_world: np.ndarray, cameras):
    """Project a 3D skeleton into all views, returning (T, V, J, 2) and (T, V, J)."""
    T = X_world.shape[0]
    V = len(cameras)
    J = X_world.shape[1]
    points_2d = np.zeros((T, V, J, 2), dtype=np.float32)
    confidences = np.ones((T, V, J), dtype=np.float32)

    for v, cam in enumerate(cameras):
        P = cam.projection_matrix
        X_h = np.concatenate([X_world, np.ones((T, J, 1))], axis=-1)  # (T, J, 4)
        x_h = np.einsum("ij,tfj->tfi", P, X_h)  # (T, J, 3)
        x = x_h[..., :2] / x_h[..., 2:3]
        points_2d[:, v, :, :] = x

    return points_2d, confidences


def test_plugin_dlt_fuse():
    """DLT backend should recover a synthetic 3D skeleton."""
    rng = np.random.default_rng(2024)
    T, J = 4, 17
    X_world = rng.uniform(-1, 1, size=(T, J, 3)) * np.array([0.5, 0.8, 1.5]) + np.array([0, 0, 3])
    cameras = _make_cameras(4)
    points_2d, confidences = _project_skeleton(X_world, cameras)

    plugin = MultiViewFusionPlugin("dlt")
    fused = plugin.fuse(points_2d, confidences, cameras)

    assert fused.shape == (T, J, 3)
    assert np.all(np.isfinite(fused))
    error = np.mean(np.linalg.norm(fused - X_world, axis=-1))
    assert error < 1e-3, f"DLT plugin reconstruction error too large: {error}"


def test_plugin_fuse_irs():
    """Plugin should produce a valid HumanMotionIR from per-view IRs."""
    rng = np.random.default_rng(2025)
    T, J = 2, 17
    X_world = rng.uniform(-1, 1, size=(T, J, 3)) + np.array([0, 0, 3])
    cameras = _make_cameras(4)
    points_2d, confidences = _project_skeleton(X_world, cameras)

    irs = []
    for v in range(len(cameras)):
        irs.append(
            HumanMotionIR(
                schema_version="1.0",
                sequence_id=f"view_{v}",
                person_id="person_0",
                fps=30.0,
                timestamps=np.arange(T, dtype=np.float64) / 30.0,
                human_model="smpl",
                pose={"transl": np.zeros((T, 3))},
                coordinate_system={
                    "up_axis": "y",
                    "forward_axis": "z",
                    "length_unit": "m",
                    "world_from_reference": np.eye(4),
                },
                per_view_2d={f"view_{v}": points_2d[:, v, :, :]},
                per_view_confidence={f"view_{v}": confidences[:, v, :]},
            )
        )

    plugin = MultiViewFusionPlugin("dlt")
    fused_ir = plugin.fuse_irs(irs, cameras, fused_sequence_id="fused_test")

    assert isinstance(fused_ir, HumanMotionIR)
    assert fused_ir.fusion_method == "dlt"
    assert fused_ir.sequence_id == "fused_test"
    assert fused_ir.pose is not None


def test_plugin_available_backends():
    """The plugin should list at least the built-in backends."""
    names = MultiViewFusionPlugin.available_backends()
    assert "dlt" in names
    assert "attention" in names


def test_plugin_factory():
    """Factory should instantiate a DLT-backed plugin."""
    plugin = create_multiview_plugin(backend="dlt")
    assert plugin.fusion_name == "dlt"


def test_plugin_from_predictions_return_ir():
    """fuse_from_predictions with return_ir=True should yield a HumanMotionIR."""
    rng = np.random.default_rng(2026)
    T, J = 2, 17
    X_world = rng.uniform(-1, 1, size=(T, J, 3)) + np.array([0, 0, 3])
    cameras = _make_cameras(4)
    points_2d, confidences = _project_skeleton(X_world, cameras)

    predictions = {
        f"view_{v}": {
            "keypoints_2d": points_2d[:, v, :, :],
            "confidence": confidences[:, v, :],
        }
        for v in range(len(cameras))
    }

    plugin = MultiViewFusionPlugin("dlt")
    fused_ir = plugin.fuse_from_predictions(
        predictions, cameras, return_ir=True, sequence_id="pred_test"
    )

    assert isinstance(fused_ir, HumanMotionIR)
    assert fused_ir.fusion_method == "dlt"
    assert "fused_joints_3d" in fused_ir.uncertainty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
