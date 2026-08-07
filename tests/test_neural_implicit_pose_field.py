"""CPU sanity test for the neural implicit 3-D pose field refiner."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.neural_implicit_pose_field import (
    NeuralImplicitPoseFieldRefiner,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_implicit_field_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointImplicitField,
)


def _make_cameras(v: int = 4):
    import numpy as np
    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).float()
    t = torch.zeros(v, 3).float()
    return K, R, t


def test_refiner_forward_backward():
    N, J, d = 2, 17, 32
    refiner = NeuralImplicitPoseFieldRefiner(
        j=J, feat_dim=d, hidden_dim=64, num_layers=2, n_iters=1, step_size=0.5
    )
    optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3)

    feat = torch.randn(N, J, d)
    pos = torch.randn(N, J, 3)
    target = torch.randn(N, J, 3)
    residual_input = torch.cat([feat, pos], dim=-1)

    delta = refiner(residual_input)
    pred = pos + delta
    loss = (pred - target).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert pred.shape == (N, J, 3)
    assert all(p.grad is not None for p in refiner.parameters() if p.requires_grad)
    print(f"refiner smoke passed (loss={loss.item():.4f})")


def test_full_model_forward_backward():
    B, T, V, J = 1, 3, 4, 17
    x = torch.randn(B, T, V, J, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0.0)
    y = torch.randn(B, T, J, 3)
    K, R, t = _make_cameras(V)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointImplicitField(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=1,
        residual_hidden=64,
        field_hidden=64,
        field_layers=2,
        field_iters=1,
        field_step_size=0.5,
        return_pp_delta=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pred, weights = model(x, K=K, R=R, t=t)
    loss = (pred - y).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    print(f"full implicit-field model smoke passed (loss={loss.item():.4f})")


def test_full_model_return_field_contract():
    B, T, V, J = 1, 2, 2, 17
    x = torch.rand(B, T, V, J, 3)
    K, R, t = _make_cameras(V)
    t[1, 0] = -1.0
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointImplicitField(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        field_hidden=16,
        field_layers=2,
        field_iters=1,
        return_pp_delta=True,
        return_field=True,
    )

    pred, weights, pp_delta, field_values = model(x, K=K, R=R, t=t)
    loss = pred.square().mean() + field_values.square().mean()
    loss.backward()

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert field_values.shape == (B, T, J)
    assert model.residual_mlp.field.field_head.weight.grad is not None


if __name__ == "__main__":
    test_refiner_forward_backward()
    test_full_model_forward_backward()
    print("All neural implicit pose field tests passed.")
