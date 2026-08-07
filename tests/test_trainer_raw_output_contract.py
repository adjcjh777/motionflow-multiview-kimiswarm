"""Raw 3D tuple slots consumed by the training entry points."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.ray_attention_hierarchical_attention_entropy_reg_model import (
    RayAttentionFusionModelHierarchicalAttentionEntropyReg,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_factorized_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_dynamic_gate_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_epipolar_dynamic_gate_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarDynamicGate,
)


def _inputs(batch=1, time=2, views=2, joints=17):
    x = torch.rand(batch, time, views, joints, 3)
    K = torch.eye(3).repeat(views, 1, 1)
    R = torch.eye(3).repeat(views, 1, 1)
    t = torch.zeros(views, 3)
    t[1, 0] = -1.0
    return x, K, R, t


def test_factorized_accepts_and_returns_raw_output():
    B, T, V, J = 1, 2, 2, 17
    x, K, R, t = _inputs(B, T, V, J)
    model = RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_view_layers=0,
        n_temporal_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        focal_max_scale=0.1,
        return_pp_delta=True,
        return_raw=True,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, focal_scale, raw = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert focal_scale.shape == (B * T, V)
    assert raw.shape == (B, T, J, 3)


def test_dynamic_gate_keeps_raw_before_gate_pair():
    B, T, V, J = 1, 2, 2, 17
    x, K, R, t = _inputs(B, T, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointDynamicGate(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_pp_delta=True,
        return_raw=True,
        return_gate=True,
    ).eval()

    with torch.no_grad():
        output = model(x, K=K, R=R, t=t)

    assert output[-3].shape == (B, T, J, 3)
    assert output[-2].shape == (B, T, V, J)
    assert output[-1].shape == (B, T, V, J)


def test_epipolar_returns_raw_in_trainer_slot():
    B, T, V, J = 1, 2, 2, 17
    x, K, R, t = _inputs(B, T, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        focal_max_scale=0.1,
        return_pp_delta=True,
        return_raw=True,
    ).eval()

    with torch.no_grad():
        output = model(x, K=K, R=R, t=t)

    assert output[-1].shape == (B, T, J, 3)


def test_epipolar_dynamic_gate_keeps_diagnostics_before_gate_pair():
    B, T, V, J = 1, 2, 2, 17
    x, K, R, t = _inputs(B, T, V, J)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarDynamicGate(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_raw=True,
        return_visibility=True,
        return_gate=True,
    ).eval()

    with torch.no_grad():
        output = model(x, K=K, R=R, t=t)

    assert output[-4].shape == (B, T, J, 3)
    assert torch.equal(output[-3], torch.ones_like(output[-3]))
    assert output[-2].shape == (B, T, V, J)
    assert output[-1].shape == (B, T, V, J)


def test_hierarchical_entropy_keeps_raw_before_scalar_loss():
    B, T, V, J = 1, 2, 2, 17
    x, K, R, t = _inputs(B, T, V, J)
    model = RayAttentionFusionModelHierarchicalAttentionEntropyReg(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_view_layers=0,
        n_temporal_layers=0,
        n_joint_graph_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_pp_delta=True,
        return_raw=True,
    ).eval()

    with torch.no_grad():
        output = model(x, K=K, R=R, t=t)

    assert output[-2].shape == (B, T, J, 3)
    assert output[-1].shape == ()
