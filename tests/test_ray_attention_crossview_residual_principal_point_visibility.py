"""Smoke test for visibility-gated cross-view PP model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)


def test_forward_and_visibility_shape():
    B, T, V, J = 2, 13, 14, 28
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
        j=J, d=64, n_views=V, n_st_layers=2, residual_hidden=128
    )
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)
    pred, weights, visibility = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert (visibility >= 0).all() and (visibility <= 1).all()


def test_all_views_dropped_fallback():
    B, T, V, J = 1, 13, 4, 17
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64, min_visible_views=2
    )
    x = torch.randn(B, T, V, J, 3)
    x[..., 2] = 0.0  # all confidences zero
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)
    pred, weights, visibility = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)


if __name__ == "__main__":
    test_forward_and_visibility_shape()
    print("Forward + visibility shape OK")
    test_all_views_dropped_fallback()
    print("Fallback guard OK")
