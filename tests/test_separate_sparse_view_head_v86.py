"""Tests for the v86 Separate Sparse-View Head."""

import pytest
import torch

from motionflow_mv.fusion.separate_sparse_view_head_v86 import SeparateSparseViewHeadV86


def _random_inputs(B=2, T=4, V=4, J=17, d=128, device="cpu"):
    """Generate random but well-formed inputs for SeparateSparseViewHeadV86."""
    tokens = torch.randn(B, T, V, J, d, device=device)
    pred_3d_init = torch.randn(B, T, J, 3, device=device)
    view_mask = torch.ones(B, T, V, dtype=torch.bool, device=device)
    return tokens, pred_3d_init, view_mask


def test_output_shape():
    tokens, pred_3d_init, view_mask = _random_inputs()
    module = SeparateSparseViewHeadV86(d=128, n_views=4, n_joints=17)
    out = module(tokens, pred_3d_init, view_mask)
    assert out.shape == (2, 4, 17, 3)


def test_identity_at_init():
    """At initialization the head should be identity (zero residual)."""
    tokens, pred_3d_init, view_mask = _random_inputs()
    module = SeparateSparseViewHeadV86(d=128, n_views=4, n_joints=17)
    out = module(tokens, pred_3d_init, view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-6)


def test_non_zero_residual_changes_output():
    """When forced to produce a non-zero residual the head changes the output."""
    tokens, pred_3d_init, view_mask = _random_inputs(B=2, T=4, V=4, J=17)
    view_mask[1, :, 2:] = False  # sparse sample for realism

    module = SeparateSparseViewHeadV86(d=128, n_views=4, n_joints=17)
    # Force a non-zero residual by overriding the gate and final layer.
    with torch.no_grad():
        module.residual_gate.fill_(1.0)
        module.mlp[-1].weight.normal_()
        module.mlp[-1].bias.normal_()

    out = module(tokens, pred_3d_init, view_mask)
    assert not torch.allclose(out, pred_3d_init, atol=1e-5)


def test_count_embedding_added():
    """When count embedding is disabled the module should still be identity."""
    tokens, pred_3d_init, view_mask = _random_inputs()
    module = SeparateSparseViewHeadV86(d=128, n_views=4, n_joints=17, use_count_embedding=False)
    out = module(tokens, pred_3d_init, view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-6)


def test_gradient_flow():
    """Gradients should flow to the sparse-view head parameters."""
    tokens, pred_3d_init, view_mask = _random_inputs(B=2, T=4, V=4, J=17)
    view_mask[:, :, 2:] = False  # 2 active views
    module = SeparateSparseViewHeadV86(d=128, n_views=4, n_joints=17)
    out = module(tokens, pred_3d_init, view_mask)
    loss = out.sum()
    loss.backward()
    assert module.mlp[-1].weight.grad is not None
    assert module.residual_gate.grad is not None
