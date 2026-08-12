"""Unit tests for motionflow_mv/fusion/view_count_conditioning_v86.py."""

from __future__ import annotations

import pytest
import torch

from motionflow_mv.fusion.view_count_conditioning_v86 import ViewCountConditioningV86


def test_forward_preserves_shape():
    """Count conditioning should preserve token shape for any active view count."""
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    view_mask[:, :, :2] = False  # 2 active views

    module = ViewCountConditioningV86(d=d, n_views=V, hidden=64, n_layers=2)
    out = module(tokens, view_mask)
    assert out.shape == tokens.shape


def test_identity_at_init():
    """At initialization the count token should be zero -> output equals input."""
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)

    module = ViewCountConditioningV86(d=d, n_views=V, hidden=64, n_layers=2)
    out = module(tokens, view_mask)
    assert torch.allclose(out, tokens, atol=1e-5)


def test_gradients_flow():
    """Gradients should reach the count embedding and MLP parameters."""
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d, requires_grad=True)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)

    module = ViewCountConditioningV86(d=d, n_views=V, hidden=64, n_layers=2)
    out = module(tokens, view_mask)
    loss = out.sum()
    loss.backward()

    assert module.count_embed.weight.grad is not None
    assert any(p.grad is not None for p in module.mlp.parameters())
    assert tokens.grad is not None


@pytest.mark.parametrize("active_views", [0, 1, 2, 3, 4])
def test_handles_all_active_view_counts(active_views: int):
    """The module should accept any active view count between 0 and n_views."""
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d)
    view_mask = torch.zeros(B, T, V, dtype=torch.bool)
    if active_views > 0:
        view_mask[:, :, :active_views] = True

    module = ViewCountConditioningV86(d=d, n_views=V, hidden=64, n_layers=2)
    out = module(tokens, view_mask)
    assert out.shape == tokens.shape
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_forward_preserves_shape()
    test_identity_at_init()
    test_gradients_flow()
    for k in range(5):
        test_handles_all_active_view_counts(k)
    print("All ViewCountConditioningV86 unit tests passed")
