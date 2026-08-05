"""Lightweight sanity tests for the temporal ray-attention fusion model."""

import sys
from pathlib import Path

import torch

# Make project root importable when running pytest directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_model import (
    RayAttentionFusionModelTemporal,
    _make_cameras,
)


def test_temporal_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporal(j=J, d=64, n_views=V, n_temporal_layers=2)
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

    model = RayAttentionFusionModelTemporal(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


if __name__ == "__main__":
    test_temporal_forward_shape_and_grad()
    test_single_frame_compatibility()
    print("temporal ray-attention tests passed")
