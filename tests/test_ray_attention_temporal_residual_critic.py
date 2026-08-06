"""Lightweight sanity tests for the temporal residual + critic model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_residual_critic_model import (
    RayAttentionFusionModelTemporalResidualCritic,
    _make_cameras,
)


def test_temporal_residual_critic_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualCritic(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_residual_critic_compatibility():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualCritic(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


if __name__ == "__main__":
    test_temporal_residual_critic_forward_shape_and_grad()
    test_single_frame_residual_critic_compatibility()
    print("temporal residual + critic refinement tests passed")
