"""CPU sanity tests for the Gaussian-splatting pose regularizer.

Run with: python -m pytest tests/test_gaussian_splatting_pose_loss.py -v
"""

import pytest
import torch

from motionflow_mv.losses.gaussian_splatting_pose_loss import (
    gaussian_splatting_pose_loss,
    gaussian_splatting_render_error,
)


def _make_cameras(B: int, V: int):
    """Build simple forward-facing cameras for testing."""
    K = torch.eye(3).unsqueeze(0).repeat(B, V, 1, 1).float()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0

    # All cameras at z = 5 m looking at origin.
    R = torch.eye(3).unsqueeze(0).repeat(B, V, 1, 1).float()
    t = torch.zeros(B, V, 3).float()
    t[..., 2] = 5.0
    return K, R, t


def test_splat_loss_forward_and_backward():
    B, T, V, J = 2, 4, 3, 17
    pred_3d = torch.randn(B, T, J, 3, requires_grad=True)
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    log_std = torch.randn(B, T, J, 3, requires_grad=True)
    confidences = torch.ones(B, T, V, J) * 0.8

    loss = gaussian_splatting_pose_loss(pred_3d, points_2d, K, R, t, log_std, confidences)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_3d.grad is not None
    assert log_std.grad is not None
    assert torch.isfinite(pred_3d.grad).all()
    assert torch.isfinite(log_std.grad).all()


def test_splat_loss_no_confidences():
    B, T, V, J = 1, 2, 2, 17
    pred_3d = torch.randn(B, T, J, 3, requires_grad=True)
    points_2d = torch.randn(B, T, V, J, 2) * 50 + 320
    K, R, t = _make_cameras(B, V)
    log_std = torch.randn(B, T, J, 3, requires_grad=True)

    loss = gaussian_splatting_pose_loss(pred_3d, points_2d, K, R, t, log_std)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_render_error_shape():
    B, T, V, J = 2, 3, 3, 17
    pred_3d = torch.randn(B, T, J, 3)
    points_2d = torch.randn(B, T, V, J, 2)
    K, R, t = _make_cameras(B, V)
    log_std = torch.randn(B, T, J, 3)

    error = gaussian_splatting_render_error(pred_3d, points_2d, K, R, t, log_std)
    assert error.shape == (B, T, V, J)
    assert torch.isfinite(error).all()


if __name__ == "__main__":
    test_splat_loss_forward_and_backward()
    test_splat_loss_no_confidences()
    test_render_error_shape()
    print("gaussian_splatting_pose_loss CPU tests passed")
