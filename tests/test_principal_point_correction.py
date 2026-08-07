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


def test_principal_point_correction_identity_initialization_and_bounds():
    N, V, J, d = 2, 4, 17, 64
    K = torch.eye(3).float().unsqueeze(0).unsqueeze(0).expand(N, V, -1, -1).contiguous()
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0

    feat = torch.randn(N, V, J, d)
    weights = torch.rand(N, V, J)

    layer = PrincipalPointCorrection(
        d=d,
        hidden=64,
        max_offset=20.0,
        max_focal_scale=0.1,
    )
    K_corr, delta, focal_scale = layer(K, feat=feat, weights=weights)

    assert K_corr.shape == K.shape
    assert delta.shape == (N, V, 2)
    assert (delta.abs() <= 20.0 + 1e-5).all()
    torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
    torch.testing.assert_close(focal_scale, torch.ones_like(focal_scale), rtol=0, atol=0)
    torch.testing.assert_close(K_corr, K, rtol=0, atol=0)

    x = torch.rand(N, V, J, 3)
    K_raw, delta_raw, focal_raw = layer(K, x=x, weights=x[..., 2])
    torch.testing.assert_close(delta_raw, torch.zeros_like(delta_raw), rtol=0, atol=0)
    torch.testing.assert_close(focal_raw, torch.ones_like(focal_raw), rtol=0, atol=0)
    torch.testing.assert_close(K_raw, K, rtol=0, atol=0)


def test_principal_point_pooling_uses_exact_final_weights():
    pool_layer = PrincipalPointCorrection(d=2, hidden=4).double()
    layer = PrincipalPointCorrection(d=8, hidden=4).double()
    with torch.no_grad():
        layer.fallback_projector.weight.copy_(torch.eye(8, dtype=torch.float64))
        layer.fallback_projector.bias.zero_()

    feat = torch.tensor(
        [[[[2.0, 20.0], [100.0, 1000.0]], [[3.0, 30.0], [200.0, 2000.0]]]],
        dtype=torch.float64,
    )
    feat_weights = torch.tensor([[[0.0, 0.0], [1e-8, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(
        pool_layer._pool_features(feat, feat_weights),
        torch.tensor([[[0.0, 0.0], [3.0, 30.0]]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )

    x = torch.tensor(
        [[
            [[2.0, 20.0, 0.0], [10.0, 100.0, 0.0]],
            [[2.0, 20.0, 1.0], [10.0, 100.0, 0.0]],
            [[2.0, 20.0, 0.5], [10.0, 100.0, 1.0]],
        ]],
        dtype=torch.float64,
    )
    one_K = torch.tensor(
        [[2.0, 0.25, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    K = one_K.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
    expected = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 3.0, 5.0, 2.0, 4.0, 0.25],
            [2.0, 20.0, 0.5, 3.0, 5.0, 2.0, 4.0, 0.25],
            [22.0 / 3.0, 220.0 / 3.0, 0.75, 3.0, 5.0, 2.0, 4.0, 0.25],
        ]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        layer._features_from_x(x, K, x[..., 2]),
        expected,
        rtol=0,
        atol=1e-12,
    )


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
    test_principal_point_correction_identity_initialization_and_bounds()
    test_principal_point_pooling_uses_exact_final_weights()
    test_principal_point_model_forward_and_grad()
    test_principal_point_model_single_frame()
    print("principal-point correction tests passed")
