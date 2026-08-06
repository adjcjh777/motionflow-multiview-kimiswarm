"""Smoke tests for the factorized ST+PP model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_factorized_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint,
)


def _make_cameras(V: int = 4):
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).repeat(V, 1, 1).float()
    t = torch.zeros(V, 3).float()
    return K, R, t


def test_factorized_pp_forward():
    B, T, V, J = 2, 13, 4, 17
    x = torch.randn(B, T, V, J, 3)
    K, R, t = _make_cameras(V)
    model = RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint(
        j=J, d=32, n_views=V, n_view_layers=1, n_temporal_layers=1, residual_hidden=64, return_pp_delta=True
    )
    pred, weights, pp_delta = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)


def test_factorized_pp_single_frame():
    V, J = 4, 17
    x = torch.randn(1, V, J, 3)
    K, R, t = _make_cameras(V)
    model = RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint(
        j=J, d=32, n_views=V, n_view_layers=1, n_temporal_layers=1, residual_hidden=64
    )
    pred, weights = model(x, K=K, R=R, t=t)
    assert pred.shape == (1, J, 3)
    assert weights.shape == (1, V, J)
