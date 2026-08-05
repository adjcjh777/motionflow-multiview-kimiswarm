"""Lightweight sanity tests for the CamPE v2 residual model."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_residual_campe_v2_model import (
    RayAttentionFusionModelTemporalResidualCamPEV2,
)
from motionflow_mv.fusion.ray_attention_temporal_residual_model import _make_cameras


def test_campe_v2_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualCamPEV2(
        j=J, d=64, n_views=V, n_temporal_layers=2
    )
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_campe_v2_single_frame():
    B, V, J = 2, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualCamPEV2(j=J, d=64, n_views=V)
    pred, weights = model(x, cameras=cameras)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)


def test_campe_v2_variable_views():
    """The same model should handle a different number of views without errors."""
    B, T, J = 2, 5, 17
    V4, V14 = 4, 14
    model = RayAttentionFusionModelTemporalResidualCamPEV2(j=J, d=64, n_views=V4)

    for V, cameras in [(V4, _make_cameras(V4)), (V14, _make_cameras(V14))]:
        x = torch.rand(B, T, V, J, 3)
        pred, weights = model(x, cameras=cameras)
        assert pred.shape == (B, T, J, 3)
        assert weights.shape == (B, T, V, J)


def test_campe_v2_n_iter():
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEV2(j=J, d=64, n_views=V)
    pred, _ = model(x, cameras=cameras, n_iter=3)
    assert pred.shape == (B, T, J, 3)


if __name__ == "__main__":
    test_campe_v2_forward_shape_and_grad()
    test_campe_v2_single_frame()
    test_campe_v2_variable_views()
    test_campe_v2_n_iter()
    print("CamPE v2 tests passed")
