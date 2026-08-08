"""CPU smoke tests for v36 Uncertainty-Gated Iterative Graph Refinement.

This module does **not** start any GPU training; it only checks that the
v36 UGIGR module can be instantiated, run a forward pass, and produce
gradients with and without a view mask.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.uncertainty_gated_iterative_graph_refinement_v36 import (
    UncertaintyGatedIterativeGraphRefinementV36,
)


def test_forward_17j_no_mask():
    B, T, V, J, d = 2, 5, 4, 17, 32
    tokens = torch.rand(B, T, V, J, d)
    model = UncertaintyGatedIterativeGraphRefinementV36(
        d=d,
        n_views=V,
        n_layers=1,
        n_iters=2,
        n_heads=2,
    )
    out = model(tokens)
    assert out.shape == (B, T, V, J, d)
    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_forward_28j_with_mask():
    B, T, V, J, d = 2, 3, 6, 28, 64
    tokens = torch.rand(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    # Mask out the last two views in the second sample.
    view_mask[1, :, -2:] = False

    model = UncertaintyGatedIterativeGraphRefinementV36(
        d=d,
        n_views=V,
        n_layers=2,
        n_iters=1,
        n_heads=4,
    )
    out = model(tokens, view_mask=view_mask)
    assert out.shape == (B, T, V, J, d)
    # Because the block is a residual addition with the refinement zeroed on
    # masked views, the output at masked views should equal the input token.
    assert torch.allclose(out[1, :, -2:, :, :], tokens[1, :, -2:, :, :], atol=1e-5)


def test_identity_at_init():
    """With zero output projection and near-zero gate, the block is ~identity."""
    B, T, V, J, d = 1, 2, 4, 17, 32
    tokens = torch.randn(B, T, V, J, d)
    model = UncertaintyGatedIterativeGraphRefinementV36(
        d=d,
        n_views=V,
        n_layers=1,
        n_iters=2,
        n_heads=2,
    )
    out = model(tokens)
    # Residual gate is sigmoid(-6) ~ 0.0025, so output should be very close
    # to input.
    assert torch.allclose(out, tokens, atol=1e-2)


if __name__ == "__main__":
    test_forward_17j_no_mask()
    print("test_forward_17j_no_mask passed")
    test_forward_28j_with_mask()
    print("test_forward_28j_with_mask passed")
    test_identity_at_init()
    print("test_identity_at_init passed")
    print("All v36 UGIGR smoke tests passed")
