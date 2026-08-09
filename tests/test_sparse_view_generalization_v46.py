"""Unit/integration tests for v46 Sparse-View Generalization (SVG).

These tests verify that:

* ``SparseViewGeneralizationV46`` predicts per-view reliability weights with the
  correct shape, positivity, and masking behaviour.
* The view-dropout augmentation drops views while respecting ``min_views``.
* ``OmniMultiViewFusionV5`` can be instantiated with the v46 flag enabled and
  runs a forward pass without crashing.

The tests are CPU-only and do not start any GPU training.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.sparse_view_generalization_v46 import (
    SparseViewGeneralizationV46,
)
from motionflow_mv.data.view_dropout_augmentation_v46 import drop_views


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras."""
    import numpy as np

    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


@pytest.fixture
def dummy_features():
    """Return random multi-view feature tokens and a full view mask."""
    B, T, V, J, C = 2, 3, 4, 17, 32
    x = torch.randn(B, T, V, J, C)
    view_mask = torch.ones(B, T, V)
    return x, view_mask


# ---------------------------------------------------------------------------
# SparseViewGeneralizationV46 unit tests
# ---------------------------------------------------------------------------


def test_module_forward_shape_and_positivity(dummy_features):
    x, view_mask = dummy_features
    B, T, V, J, C = x.shape
    module = SparseViewGeneralizationV46(in_channels=C, n_views=V, hidden=16)
    reliability = module(x, view_mask=view_mask)
    assert reliability.shape == (B, T, V, J)
    assert reliability.min().item() > 0.0
    assert reliability.max().item() < 2.0


def test_module_default_mask_is_all_ones(dummy_features):
    x, _ = dummy_features
    module = SparseViewGeneralizationV46(in_channels=x.shape[-1], n_views=x.shape[2])
    reliability = module(x)
    assert reliability.shape == x.shape[:-1]


def test_module_masks_dropped_views(dummy_features):
    x, view_mask = dummy_features
    B, T, V, J, C = x.shape
    module = SparseViewGeneralizationV46(in_channels=C, n_views=V, hidden=16)
    view_mask[:, :, -1] = 0.0
    reliability = module(x, view_mask=view_mask)
    # Active views still receive positive weights.
    assert reliability[..., :-1, :].max().item() > 0.0
    # Dropped view is zeroed.
    assert reliability[..., -1, :].max().item() == 0.0


def test_module_initial_weights_near_one(dummy_features):
    """Zero-initialised final layer -> 2 * sigmoid(0) = 1.0."""
    x, _ = dummy_features
    module = SparseViewGeneralizationV46(
        in_channels=x.shape[-1], n_views=x.shape[2], hidden=16
    )
    reliability = module(x)
    assert torch.allclose(reliability, torch.ones_like(reliability), atol=0.2)


def test_module_gradients_flow(dummy_features):
    x, view_mask = dummy_features
    module = SparseViewGeneralizationV46(
        in_channels=x.shape[-1], n_views=x.shape[2], hidden=8
    )
    reliability = module(x, view_mask=view_mask)
    loss = reliability.sum()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())


def test_module_accepts_two_dimensional_view_mask(dummy_features):
    x, _ = dummy_features
    B, T, V, J, C = x.shape
    module = SparseViewGeneralizationV46(in_channels=C, n_views=V, hidden=8)
    view_mask = torch.ones(B, V)
    view_mask[:, -1] = 0.0
    reliability = module(x, view_mask=view_mask)
    assert reliability.shape == (B, T, V, J)
    assert reliability[:, :, -1, :].max().item() == 0.0


# ---------------------------------------------------------------------------
# View dropout augmentation tests
# ---------------------------------------------------------------------------


def _make_views_with_confidence(B, T, V, J):
    """Create a view tensor with shape ``(B, T, V, J, 3)`` where the last
    channel is a confidence map.
    """
    views = torch.randn(B, T, V, J, 3)
    views[..., 2] = torch.rand(B, T, V, J)
    return views


def test_drop_views_respects_min_views():
    """Ensure at least ``min_views`` remain after dropout."""
    B, T, V, J = 2, 3, 4, 17
    views = _make_views_with_confidence(B, T, V, J)
    aug_views, view_mask = drop_views(views, prob=1.0, min_views=2)
    assert aug_views.shape == views.shape
    assert view_mask.shape == (B, V)
    assert view_mask.sum(dim=-1).min().item() >= 2


def test_drop_views_zero_prob_keeps_all_views():
    B, T, V, J = 2, 3, 4, 17
    views = _make_views_with_confidence(B, T, V, J)
    aug_views, view_mask = drop_views(views, prob=0.0, min_views=2)
    assert torch.allclose(view_mask, torch.ones_like(view_mask))


def test_drop_views_zeroes_confidence_channel_of_dropped_views():
    B, T, V, J = 2, 3, 4, 17
    views = _make_views_with_confidence(B, T, V, J)
    # Force at least one view to be dropped by setting prob high and using a seed.
    aug_views, view_mask = drop_views(views, prob=0.9, min_views=1, seed=42)
    assert aug_views.shape == views.shape
    for b in range(B):
        for v in range(V):
            if view_mask[b, v].item() == 0.0:
                assert aug_views[b, :, v, :, 2].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_v46_flag_wires_into_omniview_fusion_v5():
    """OmniMultiViewFusionV5 with v46 enabled can run a forward pass."""
    from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5

    B, T, V, J = 2, 3, 4, 17
    K, R, t = _make_cameras(V)
    # OmniMultiViewFusionV5 expects (B, T, V, J, 3) where channel 2 is confidence.
    x = torch.cat([torch.randn(B, T, V, J, 2), torch.ones(B, T, V, J, 1)], dim=-1)
    model = OmniMultiViewFusionV5(
        j=J,
        d=32,
        n_views=V,
        n_heads=2,
        n_st_layers=1,
        use_multiview_geometry_fusion_v25=True,
        use_v46_sparse_view_generalization=True,
        v46_svg_hidden=16,
        return_covariance=False,
    )
    pred_3d, weights, visibility, L, epi_loss = model(
        x=x,
        K=K.unsqueeze(0).expand(B, -1, -1, -1),
        R=R.unsqueeze(0).expand(B, -1, -1, -1),
        t=t.unsqueeze(0).expand(B, -1, -1),
    )
    assert pred_3d.shape == (B, T, J, 3)
    assert epi_loss.numel() == 1
