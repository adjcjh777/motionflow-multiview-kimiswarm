"""CPU smoke test for the combined visibility + uncertainty v1 model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.models.crossview_residual_visibility_uncertainty_v1 import (
    CrossviewResidualVisibilityUncertaintyV1,
)


def test_forward_shapes_and_ranges():
    B, T, V, J = 2, 13, 14, 28
    model = CrossviewResidualVisibilityUncertaintyV1(
        j=J,
        d=64,
        n_views=V,
        n_st_layers=2,
        residual_hidden=128,
    )
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)

    pred, weights, visibility, log_var, nll_loss = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert log_var.shape == (B, T, V, J)
    assert nll_loss.shape == ()
    assert (visibility >= 0).all() and (visibility <= 1).all()


def test_gradients_flow():
    B, T, V, J = 1, 5, 4, 17
    model = CrossviewResidualVisibilityUncertaintyV1(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=1,
        residual_hidden=64,
    )
    x = torch.randn(B, T, V, J, 3, requires_grad=False)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)

    pred, *_ = model(x, K=K, R=R, t=t)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_single_frame_input():
    B, V, J = 2, 4, 17
    model = CrossviewResidualVisibilityUncertaintyV1(
        j=J,
        d=32,
        n_views=V,
        n_st_layers=1,
        residual_hidden=64,
    )
    x = torch.randn(B, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.randn(V, 3)

    pred, weights, visibility, log_var, nll_loss = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert visibility.shape == (B, V, J)
    assert log_var.shape == (B, V, J)


if __name__ == "__main__":
    test_forward_shapes_and_ranges()
    print("Forward shapes and ranges OK")
    test_gradients_flow()
    print("Gradient flow OK")
    test_single_frame_input()
    print("Single-frame input OK")
