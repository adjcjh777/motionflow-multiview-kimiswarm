"""Tests for deformable cross-view attention module."""

import pytest
import torch

from motionflow_mv.fusion.deformable_cross_view_attention import (
    DeformableCrossViewAttention,
)


def _make_dummy_cameras(batch_size: int, n_views: int):
    """Create identity-like intrinsics/extrinsics for a batch."""
    K = torch.eye(3).unsqueeze(0).expand(n_views, -1, -1).clone()
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    K = K.unsqueeze(0).expand(batch_size, -1, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(batch_size, n_views, -1, -1)
    t = torch.zeros(batch_size, n_views, 3)
    return K, R, t


def test_deformable_attention_shape_and_gradient():
    B, T, V, J, d = 2, 3, 4, 17, 64
    x = torch.randn(B, T, V, J, d, requires_grad=True)
    N = B * T
    K, R, t = _make_dummy_cameras(N, V)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    module = DeformableCrossViewAttention(
        d=d, n_heads=4, n_views=V, n_samples=2
    )
    out = module(x, K, R, t, points_2d)
    assert out.shape == (B, T, V, J, d)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in module.parameters())


def test_deformable_attention_view_mask_zeros_out_masked_views():
    B, T, V, J, d = 2, 3, 4, 17, 64
    x = torch.randn(B, T, V, J, d, requires_grad=True)
    N = B * T
    K, R, t = _make_dummy_cameras(N, V)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    module = DeformableCrossViewAttention(
        d=d, n_heads=4, n_views=V, n_samples=2
    )

    view_mask = torch.ones(B, T, V)
    view_mask[:, :, 3] = 0.0

    out = module(x, K, R, t, points_2d, view_mask=view_mask)
    assert out.shape == (B, T, V, J, d)
    # The masked-out view should not contribute to the output.
    assert out[:, :, 3, :, :].abs().max().item() < 1e-5


def test_deformable_attention_n_samples_one():
    B, T, V, J, d = 1, 2, 4, 17, 64
    x = torch.randn(B, T, V, J, d, requires_grad=True)
    N = B * T
    K, R, t = _make_dummy_cameras(N, V)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    module = DeformableCrossViewAttention(
        d=d, n_heads=4, n_views=V, n_samples=1
    )
    out = module(x, K, R, t, points_2d)
    assert out.shape == (B, T, V, J, d)


def test_deformable_attention_variable_view_mask_2d():
    B, T, V, J, d = 2, 3, 4, 17, 64
    x = torch.randn(B, T, V, J, d, requires_grad=True)
    N = B * T
    K, R, t = _make_dummy_cameras(N, V)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    module = DeformableCrossViewAttention(
        d=d, n_heads=4, n_views=V, n_samples=2
    )
    view_mask = torch.ones(B, V)
    view_mask[:, 0] = 0.0
    out = module(x, K, R, t, points_2d, view_mask=view_mask)
    assert out.shape == (B, T, V, J, d)
    assert out[:, :, 0, :, :].abs().max().item() < 1e-5


def test_deformable_attention_topk_straight_through():
    """Straight-through top-k mode should preserve shape and gradients."""
    B, T, V, J, d = 2, 3, 4, 17, 64
    x = torch.randn(B, T, V, J, d, requires_grad=True)
    N = B * T
    K, R, t = _make_dummy_cameras(N, V)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    module = DeformableCrossViewAttention(
        d=d, n_heads=4, n_views=V, n_samples=2, use_topk_straight_through=True
    )
    out = module(x, K, R, t, points_2d)
    assert out.shape == (B, T, V, J, d)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in module.parameters())


def test_deformable_attention_invalid_head_dim():
    with pytest.raises(ValueError):
        DeformableCrossViewAttention(d=64, n_heads=3, n_views=4, n_samples=2)
