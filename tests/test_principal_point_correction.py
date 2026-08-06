"""Lightweight sanity tests for the principal-point correction layer and model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.principal_point_correction import PrincipalPointCorrection
from motionflow_mv.fusion.ray_attention_temporal_residual_principal_point_model import (
    RayAttentionFusionModelTemporalResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_model import _make_cameras


def test_principal_point_correction_bounds():
    N, V, J, d = 2, 4, 17, 64
    K = torch.eye(3).float().unsqueeze(0).unsqueeze(0).expand(N, V, -1, -1).contiguous()
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0

    feat = torch.randn(N, V, J, d)
    weights = torch.rand(N, V, J)

    layer = PrincipalPointCorrection(d=d, hidden=64, max_offset=20.0)
    K_corr, delta = layer(K, feat=feat, weights=weights)

    assert K_corr.shape == K.shape
    assert delta.shape == (N, V, 2)
    assert (delta.abs() <= 20.0 + 1e-5).all()
    # Near-identity at init (allow a few pixels due to random init).
    assert delta.abs().mean() < 5.0


def test_principal_point_model_forward_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualPrincipalPoint(
        j=J, d=64, n_views=V, principal_point_max_offset=20.0
    )
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_principal_point_model_single_frame():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualPrincipalPoint(
        j=J, d=64, n_views=V, principal_point_max_offset=20.0
    )
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, J, 3)
    assert w.shape == (B, V, J)


if __name__ == "__main__":
    test_principal_point_correction_bounds()
    test_principal_point_model_forward_and_grad()
    test_principal_point_model_single_frame()
    print("principal-point correction tests passed")
