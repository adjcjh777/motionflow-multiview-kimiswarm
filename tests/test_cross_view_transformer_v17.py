"""Tests for ``motionflow_mv.fusion.cross_view_transformer_v17``."""

import pytest
import torch

from motionflow_mv.fusion.cross_view_transformer_v17 import CrossViewTransformerV17


def _random_cameras(batch_size: int, n_views: int) -> tuple:
    """Create random but well-formed intrinsics/extrinsics."""
    K = torch.eye(3).view(1, 1, 3, 3).expand(batch_size, n_views, 3, 3).clone()
    K[:, :, 0, 0] = 800.0
    K[:, :, 1, 1] = 800.0
    K[:, :, 0, 2] = 320.0
    K[:, :, 1, 2] = 240.0

    # Random rotation matrices via QR.
    rand = torch.randn(batch_size, n_views, 3, 3)
    R, _ = torch.linalg.qr(rand)

    t = torch.randn(batch_size, n_views, 3)
    return K, R, t


@pytest.mark.parametrize("n_layers", [1, 2])
def test_cross_view_transformer_v17_forward_shape(n_layers: int) -> None:
    B, T, V, J, d = 2, 5, 4, 17, 64
    x = torch.randn(B, T, V, J, d)
    points_2d = torch.randn(B, T, V, J, 2)
    K, R, t = _random_cameras(B * T, V)

    module = CrossViewTransformerV17(d=d, n_heads=4, n_layers=n_layers, dropout=0.0)
    out = module(x, K=K, R=R, t=t, points_2d=points_2d)

    assert out.shape == (B, T, V, J, d)


def test_cross_view_transformer_v17_no_geometry() -> None:
    """Module works as a vanilla view transformer when no cameras are supplied."""
    B, T, V, J, d = 2, 3, 4, 17, 32
    x = torch.randn(B, T, V, J, d)

    module = CrossViewTransformerV17(d=d, n_heads=4, n_layers=1)
    out = module(x)

    assert out.shape == (B, T, V, J, d)


def test_cross_view_transformer_v17_view_mask_zeroes() -> None:
    """Masked-out views are zeroed in the output."""
    B, T, V, J, d = 2, 3, 4, 17, 32
    x = torch.randn(B, T, V, J, d)
    points_2d = torch.randn(B, T, V, J, 2)
    K, R, t = _random_cameras(B * T, V)

    view_mask = torch.ones(B, T, V)
    view_mask[:, :, 2:] = 0.0

    module = CrossViewTransformerV17(d=d, n_heads=4, n_layers=1)
    out = module(x, K=K, R=R, t=t, points_2d=points_2d, view_mask=view_mask)

    assert out.shape == (B, T, V, J, d)
    assert out[:, :, 2:, :, :].abs().max().item() < 1e-6


def test_cross_view_transformer_v17_gradients() -> None:
    B, T, V, J, d = 2, 3, 4, 17, 32
    x = torch.randn(B, T, V, J, d)
    points_2d = torch.randn(B, T, V, J, 2)
    K, R, t = _random_cameras(B * T, V)

    module = CrossViewTransformerV17(d=d, n_heads=4, n_layers=2)
    out = module(x, K=K, R=R, t=t, points_2d=points_2d)
    loss = out.sum()
    loss.backward()

    assert any(p.grad is not None for p in module.parameters())
