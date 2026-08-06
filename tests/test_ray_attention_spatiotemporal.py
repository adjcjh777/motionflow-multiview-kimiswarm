"""Sanity tests for the spatio-temporal ray-aware attention fusion model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_spatiotemporal_model import (
    RayAttentionFusionModelSpatiotemporal,
    _make_cameras,
)


def test_spatiotemporal_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelSpatiotemporal(j=J, d=64, n_views=V, n_st_layers=2)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_compatibility():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelSpatiotemporal(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_per_sample_rig():
    B, T, V, J = 2, 5, 4, 17
    x = torch.rand(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.zeros(V, 3)

    model = RayAttentionFusionModelSpatiotemporal(j=J, d=64, n_views=V)
    pred, weights = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)


if __name__ == "__main__":
    test_spatiotemporal_forward_shape_and_grad()
    print("spatio-temporal forward + grad test passed")
    test_single_frame_compatibility()
    print("single-frame compatibility test passed")
    test_per_sample_rig()
    print("per-sample rig test passed")
    print("all spatio-temporal ray-attention tests passed")
