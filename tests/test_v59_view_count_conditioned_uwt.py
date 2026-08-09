"""Minimal unit test for v59 View-Count-Conditioned Sparse-View Reliability."""

import torch

from motionflow_mv.fusion.view_count_conditioned_reliability_v59 import (
    ViewCountConditionedReliabilityV59,
)


def test_v59_identity_at_init() -> None:
    module = ViewCountConditionedReliabilityV59(d=64, n_views=4, hidden=32, n_layers=2, max_views=8)
    B, T, V, J = 2, 3, 4, 17
    features = torch.randn(B, T, V, J, 64)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    offset = module(features, view_mask)
    assert offset.shape == (B, T, V, J)
    assert torch.allclose(offset, torch.zeros_like(offset), atol=1e-6)


def test_v59_output_shape_and_count_sensitivity() -> None:
    module = ViewCountConditionedReliabilityV59(d=32, n_views=3, hidden=16, n_layers=2, max_views=8)
    B, T, V, J = 1, 2, 3, 10
    features = torch.randn(B, T, V, J, 32)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    view_mask[0, 0, 2] = False
    offset = module(features, view_mask)
    assert offset.shape == (B, T, V, J)
    assert torch.isfinite(offset).all()
