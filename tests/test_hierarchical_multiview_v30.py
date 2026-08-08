"""Smoke tests for HierarchicalViewEncoderV30."""

import pytest
import torch

from motionflow_mv.fusion.hierarchical_multiview_v30 import HierarchicalViewEncoderV30


@pytest.mark.parametrize("j", [17, 28, 12])
def test_hierarchical_v30_identity_at_init(j: int) -> None:
    """With the residual gate near zero, the block should be close to identity."""
    m = HierarchicalViewEncoderV30(d=64, n_heads=4, n_views=4, n_part_layers=2)
    x = torch.randn(2, 3, 4, j, 64)
    y = m(x)
    # Gate is initialized to -6 -> sigmoid ~0.0025, so output ~ input.
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-2)


def test_hierarchical_v30_view_mask() -> None:
    """Masked views should not contribute to the output."""
    m = HierarchicalViewEncoderV30(d=64, n_heads=4, n_views=4)
    x = torch.randn(2, 3, 4, 17, 64)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, :, -1] = False
    y = m(x, view_mask=mask)
    assert y.shape == x.shape
