"""Smoke tests for the occlusion-robust visibility transformer model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_transformer_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer,
)


def test_forward_and_visibility_logits_shape():
    B, T, V, J = 2, 13, 14, 28
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
        j=J, d=64, n_views=V, n_st_layers=2, residual_hidden=128
    )
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)
    pred, weights, visibility, logits = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert logits.shape == (B, T, V, J)
    assert (visibility >= 0).all() and (visibility <= 1).all()


def test_gradient_flow():
    B, T, V, J = 1, 3, 4, 17
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64
    )
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)
    pred, weights, visibility, logits = model(x, K=K, R=R, t=t)
    loss = pred.mean() + logits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_input():
    B, V, J = 2, 4, 17
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64
    )
    x = torch.randn(B, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)
    pred, weights, visibility, logits = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert visibility.shape == (B, V, J)
    assert logits.shape == (B, V, J)


if __name__ == "__main__":
    test_forward_and_visibility_logits_shape()
    print("Forward + visibility logits shape OK")
    test_gradient_flow()
    print("Gradient flow OK")
    test_single_frame_input()
    print("Single-frame input OK")
