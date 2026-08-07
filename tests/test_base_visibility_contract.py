"""Output contract for the base model's identity visibility gate."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.action_aware_principal_point_model import (
    ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def test_base_visibility_matches_input_layout():
    batch, views, joints = 1, 2, 17
    x = torch.randn(batch, views, joints, 3)
    x[..., 2] = 1.0
    K = torch.eye(3).repeat(views, 1, 1)
    R = torch.eye(3).repeat(views, 1, 1)
    t = torch.zeros(views, 3)
    t[1, 0] = -1.0
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=joints,
        d=16,
        n_views=views,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_visibility=True,
    ).eval()

    with torch.no_grad():
        single = model(x, K=K, R=R, t=t)
        sequence = model(x[:, None], K=K, R=R, t=t)

    assert single[2].shape == (batch, views, joints)
    assert sequence[2].shape == (batch, 1, views, joints)
    assert torch.equal(single[2], torch.ones_like(single[2]))
    assert torch.equal(sequence[2], torch.ones_like(sequence[2]))


def test_action_aware_returns_visibility_after_raw():
    batch, views, joints = 1, 2, 17
    x = torch.randn(batch, views, joints, 3)
    x[..., 2] = 1.0
    K = torch.eye(3).repeat(views, 1, 1)
    R = torch.eye(3).repeat(views, 1, 1)
    t = torch.zeros(views, 3)
    t[1, 0] = -1.0
    model = ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=joints,
        d=16,
        n_views=views,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        return_raw=True,
        return_visibility=True,
    ).eval()

    with torch.no_grad():
        _, _, raw, visibility = model(x, K=K, R=R, t=t)

    assert raw.shape == (batch, joints, 3)
    assert torch.equal(visibility, torch.ones(batch, views, joints))
