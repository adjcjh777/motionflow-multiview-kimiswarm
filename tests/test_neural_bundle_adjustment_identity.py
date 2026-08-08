"""Tests for NeuralBundleAdjustment identity initialization.

The neural camera-correction head is initialized so that it predicts zero
corrections at start of training.  This means a freshly-initialized module
should leave the input camera parameters (intrinsics, rotations and
translations) essentially unchanged, while the analytic 3D point update may
still refine the skeleton.
"""

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


def test_identity_initialization_leaves_cameras_unchanged():
    """Fresh module should not perturb K, R, t because the final MLP layer is zero."""
    B, T, V, J = 2, 3, 4, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=2, camera_hidden=32)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)

    # Cameras must stay identical to initialization (identity camera head).
    assert torch.allclose(K_ref, K, atol=1e-6)
    assert torch.allclose(R_ref, R, atol=1e-6)
    assert torch.allclose(t_ref, t, atol=1e-6)

    # Shapes should still match.
    assert X_ref.shape == X.shape
    assert K_ref.shape == K.shape
    assert R_ref.shape == R.shape
    assert t_ref.shape == t.shape


def test_identity_initialization_no_temporal_dim():
    """Same identity property when the temporal dimension is omitted."""
    B, V, J = 2, 4, 17
    X = torch.randn(B, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, V, J)

    nba = NeuralBundleAdjustment(n_iters=1, camera_hidden=16)
    _, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)

    assert torch.allclose(K_ref, K, atol=1e-6)
    assert torch.allclose(R_ref, R, atol=1e-6)
    assert torch.allclose(t_ref, t, atol=1e-6)


def test_identity_initialization_preserves_camera_gradients():
    """Even though corrections are zero at init, gradients should still flow."""
    B, T, V, J = 2, 2, 3, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K, R, t = _make_cameras(B, V)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=1, camera_hidden=16)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)
    loss = X_ref.mean() + K_ref.mean() + R_ref.mean() + t_ref.mean()
    loss.backward()

    # The zero-initialized final layer still has non-zero gradients flowing
    # through the MLP, and the camera-head parameters should receive gradients.
    assert any(p.grad is not None for p in nba.camera_head.parameters())
