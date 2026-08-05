"""Lightweight sanity tests for the uncertainty-aware residual refinement model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import _make_cameras
from motionflow_mv.fusion.ray_attention_temporal_residual_v3_model import RayAttentionFusionModelTemporalResidualV3


def test_temporal_residual_v3_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualV3(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_residual_v3_compatibility():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualV3(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_uncertainty_positive_and_finite():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualV3(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    # DLT weights should remain non-negative and finite.
    assert (weights >= 0).all()
    assert torch.isfinite(pred).all()


if __name__ == "__main__":
    test_temporal_residual_v3_forward_shape_and_grad()
    test_single_frame_residual_v3_compatibility()
    test_uncertainty_positive_and_finite()
    print("temporal residual v3 uncertainty tests passed")
