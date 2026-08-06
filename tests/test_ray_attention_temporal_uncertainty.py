"""Lightweight sanity tests for the uncertainty-weighted temporal model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_uncertainty_model import (
    RayAttentionFusionModelTemporalUncertainty,
)


def test_uncertainty_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    x = torch.rand(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    t = torch.zeros(V, 3)

    model = RayAttentionFusionModelTemporalUncertainty(
        j=J, d=64, n_views=V, n_temporal_layers=2
    )
    pred, weights, log_var, nll = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert log_var.shape == (B, T, V, J)
    assert nll.ndim == 0

    (pred.mean() + nll).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_compatibility():
    B, V, J = 2, 4, 17
    x = torch.rand(B, V, J, 3)
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    t = torch.zeros(V, 3)

    model = RayAttentionFusionModelTemporalUncertainty(
        j=J, d=64, n_views=V
    )
    pred, weights, log_var, nll = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert log_var.shape == (B, V, J)


if __name__ == "__main__":
    test_uncertainty_forward_shape_and_grad()
    test_single_frame_compatibility()
    print("uncertainty-weighted temporal tests passed")
