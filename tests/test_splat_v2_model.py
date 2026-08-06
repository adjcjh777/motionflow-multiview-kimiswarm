"""CPU sanity tests for the view-dependent Gaussian-splatting fusion model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_splat_v2_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplatV2,
)
from motionflow_mv.losses.gaussian_splatting_pose_loss import gaussian_splatting_pose_loss


def _make_cameras(B: int, V: int):
    K = torch.eye(3).unsqueeze(0).repeat(B, V, 1, 1).float()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).repeat(B, V, 1, 1).float()
    t = torch.zeros(B, V, 3).float()
    t[..., 2] = 5.0
    return K, R, t


def test_splat_v2_forward_backward():
    B, T, V, J = 2, 4, 3, 17
    x = torch.randn(B, T, V, J, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0)
    y = torch.randn(B, T, J, 3)
    K, R, t = _make_cameras(B, V)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSplatV2(
        j=J, d=32, n_views=V, n_st_layers=1, residual_hidden=64,
        return_pp_delta=True, return_covariance=True, return_view_covariance=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred, weights, pp_delta, log_std_world, log_std_view = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert log_std_world.shape == (B, T, J, 3)
    assert log_std_view.shape == (B, T, V, J, 3)

    loss = (pred - y).pow(2).mean()
    loss_splat = gaussian_splatting_pose_loss(
        pred, x[..., :2], K, R, t, log_std_world,
        confidences=x[..., 2], log_std_view=log_std_view,
    )
    total_loss = loss + 0.05 * loss_splat
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # The covariance heads used in the splat loss must receive gradients.
    assert model.covariance_head[-1].weight.grad is not None
    assert model.view_covariance_head[-1].weight.grad is not None


def test_splat_v2_loss_no_view_residual():
    """Loss should be backward compatible when no per-view residual is passed."""
    B, T, V, J = 2, 3, 3, 17
    pred = torch.randn(B, T, J, 3, requires_grad=True)
    x = torch.randn(B, T, V, J, 3)
    K, R, t = _make_cameras(B, V)
    log_std = torch.randn(B, T, J, 3, requires_grad=True)

    loss = gaussian_splatting_pose_loss(pred, x[..., :2], K, R, t, log_std, confidences=x[..., 2])
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert log_std.grad is not None


if __name__ == "__main__":
    test_splat_v2_forward_backward()
    test_splat_v2_loss_no_view_residual()
    print("SplatV2 model and loss tests passed")
