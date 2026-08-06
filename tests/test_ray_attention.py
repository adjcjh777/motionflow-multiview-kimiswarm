"""Unit tests for the ray-aware attention fusion model."""

import numpy as np
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel
from motionflow_mv.fusion.ray_attention_module import RayAttentionFusionModule


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


def test_ray_attention_model_shape():
    cameras = _make_cameras(4)
    model = RayAttentionFusionModel(j=17, d=64, n_views=4)
    x = torch.rand(2, 4, 17, 3)
    pred, weights = model(x, cameras)
    assert pred.shape == (2, 17, 3)
    assert weights.shape == (2, 4, 17)


def test_ray_attention_module_shape():
    cameras = _make_cameras(4)
    module = RayAttentionFusionModule(j=17, d=64, n_views=4)
    points_2d = np.random.randn(2, 4, 17, 2).astype(np.float32)
    confidences = np.random.rand(2, 4, 17).astype(np.float32)
    pred = module.fuse(points_2d, confidences, cameras)
    assert pred.shape == (2, 17, 3)


if __name__ == "__main__":
    test_ray_attention_model_shape()
    test_ray_attention_module_shape()
    print("ray_attention tests passed")
