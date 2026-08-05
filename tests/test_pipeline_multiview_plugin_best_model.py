"""Smoke test for the best cross-view PP model as a FusionModule plugin."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.pipeline_multiview_plugin import MultiViewFusionPlugin


def _make_circular_cameras(n_views=4):
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / (np.linalg.norm(c) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t_vec = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t_vec))
    return cameras


def test_plugin_backend_registered():
    names = MultiViewFusionPlugin.available_backends()
    assert "ray_attention_temporal_crossview_residual_principal_point" in names, names


def test_plugin_forward_shape():
    n_views, j, t = 4, 17, 9
    cameras = _make_circular_cameras(n_views)
    plugin = MultiViewFusionPlugin(
        fusion_name="ray_attention_temporal_crossview_residual_principal_point",
        device="cpu",
    )

    points_2d = np.random.randn(t, n_views, j, 2).astype(np.float32) * 0.1
    confidences = np.ones((t, n_views, j), dtype=np.float32)

    fused = plugin.fuse(points_2d, confidences, cameras)
    assert fused.shape == (t, j, 3)


if __name__ == "__main__":
    test_plugin_backend_registered()
    print("Backend registered OK")
    test_plugin_forward_shape()
    print("Plugin forward shape OK")
