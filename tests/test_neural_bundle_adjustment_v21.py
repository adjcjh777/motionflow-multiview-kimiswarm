"""Tests for the v21 neural bundle-adjustment layer."""

import torch

from motionflow_mv.fusion.neural_bundle_adjustment_v21 import NeuralBundleAdjustment


def _make_cameras(batch: int, n_views: int):
    """Create simple forward-facing intrinsics and identity rotations."""
    K = torch.eye(3).view(1, 1, 3, 3).expand(batch, n_views, 3, 3).clone()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    R = torch.eye(3).view(1, 1, 3, 3).expand(batch, n_views, 3, 3).clone()
    t = torch.zeros(batch, n_views, 3)
    return K, R, t


def test_neural_bundle_adjustment_forward_shape():
    B, T, V, J = 2, 3, 4, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=2, camera_hidden=32)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)

    assert X_ref.shape == X.shape
    assert K_ref.shape == K.shape
    assert R_ref.shape == R.shape
    assert t_ref.shape == t.shape


def test_neural_bundle_adjustment_no_temporal_dim():
    B, V, J = 2, 4, 17
    X = torch.randn(B, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, V, J)

    nba = NeuralBundleAdjustment(n_iters=1, camera_hidden=16)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)

    assert X_ref.shape == X.shape
    assert K_ref.shape == K.shape
    assert R_ref.shape == R.shape
    assert t_ref.shape == t.shape


def test_neural_bundle_adjustment_rotations_stay_orthogonal():
    B, T, V, J = 1, 1, 4, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=2, camera_hidden=16)
    _, _, R_ref, _ = nba(X, points_2d, K, R, t, weights)

    identity = torch.matmul(R_ref, R_ref.transpose(-2, -1))
    eye = torch.eye(3, device=R_ref.device, dtype=R_ref.dtype)
    assert torch.allclose(identity, eye.expand_as(identity), atol=1e-4)


def test_neural_bundle_adjustment_backward():
    B, T, V, J = 2, 2, 3, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=1, camera_hidden=16)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)
    loss = X_ref.mean() + K_ref.mean() + R_ref.mean() + t_ref.mean()
    loss.backward()

    assert any(p.grad is not None for p in nba.parameters())
