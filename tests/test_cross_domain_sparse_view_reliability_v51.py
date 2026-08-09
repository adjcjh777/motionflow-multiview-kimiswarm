"""Unit tests for v51 Cross-Domain Sparse-View Reliability (CDSVR)."""

from __future__ import annotations

import pytest
import torch

from motionflow_mv.fusion.cross_domain_sparse_view_reliability_v51 import (
    CrossDomainSparseViewReliabilityV51,
)


@pytest.fixture
def shapes():
    B, V, J = 3, 4, 17
    return B, V, J


def test_forward_shape_with_domain_id(shapes):
    B, V, J = shapes
    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    offset, scale = module(reliability, log_var, domain_id=domain_id)
    assert offset.shape == (B, V)
    assert scale.shape == (B, J)


def test_forward_shape_with_domain_emb(shapes):
    B, V, J = shapes
    hidden = 32
    module = CrossDomainSparseViewReliabilityV51(
        n_views=V, n_joints=J, hidden=hidden
    )

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_emb = torch.randn(B, hidden)

    offset, scale = module(reliability, log_var, domain_emb=domain_emb)
    assert offset.shape == (B, V)
    assert scale.shape == (B, J)


def test_identity_at_init(shapes):
    B, V, J = shapes
    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    offset, scale = module(reliability, log_var, domain_id=domain_id)
    torch.testing.assert_close(offset, torch.zeros_like(offset), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(scale, torch.ones_like(scale), atol=1e-6, rtol=1e-6)


def test_offset_clamped(shapes):
    B, V, J = shapes
    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    offset, _ = module(reliability, log_var, domain_id=domain_id)
    assert (offset.abs() <= 2.0 + 1e-6).all()


def test_scale_positive_and_clamped(shapes):
    B, V, J = shapes
    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    _, scale = module(reliability, log_var, domain_id=domain_id)
    assert (scale >= 1e-3).all()
    assert (scale <= 10.0 + 1e-6).all()


def test_backward(shapes):
    B, V, J = shapes
    module = CrossDomainSparseViewReliabilityV51(n_views=V, n_joints=J)

    reliability = torch.rand(B, V)
    log_var = torch.randn(B, J)
    domain_id = torch.randint(0, 6, (B,))

    offset, scale = module(reliability, log_var, domain_id=domain_id)
    loss = offset.sum() + scale.sum()
    loss.backward()

    for p in module.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
