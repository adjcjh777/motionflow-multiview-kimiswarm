"""Forward/backward sanity checks for RayAttentionFusionModelV4Residual."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_v4_residual_model import RayAttentionFusionModelV4Residual


def _make_cameras(n_views: int = 4) -> list:
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


def test_v4_residual_shape():
    cameras = _make_cameras(4)
    model = RayAttentionFusionModelV4Residual(j=17, d=64, n_views=4)
    x = torch.rand(2, 4, 17, 3)
    pred, weights = model(x, cameras)
    assert pred.shape == (2, 17, 3)
    assert weights.shape == (2, 4, 17)


def test_v4_residual_gradient():
    cameras = _make_cameras(4)
    model = RayAttentionFusionModelV4Residual(j=17, d=64, n_views=4)
    x = torch.rand(2, 4, 17, 3)
    pred, _ = model(x, cameras)
    pred.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_v4_residual_14_view_shape():
    cameras = _make_cameras(14)
    model = RayAttentionFusionModelV4Residual(j=28, d=64, n_views=14)
    x = torch.rand(2, 14, 28, 3)
    pred, weights = model(x, cameras)
    assert pred.shape == (2, 28, 3)
    assert weights.shape == (2, 14, 28)


if __name__ == "__main__":
    test_v4_residual_shape()
    test_v4_residual_gradient()
    test_v4_residual_14_view_shape()
    print("v4 residual tests passed")
