"""Unit tests for v46 view-dropout augmentation helper."""

import torch
import pytest

from motionflow_mv.data.view_dropout_augmentation_v46 import (
    drop_views,
    ViewDropoutAugmentationV46,
)


@pytest.fixture
def sample_views():
    B, T, V, J, C = 2, 5, 4, 17, 3
    views = torch.rand(B, T, V, J, C)
    views[..., 2] = torch.rand(B, T, V, J)
    return views


def test_drop_views_shape(sample_views):
    B = sample_views.shape[0]
    V = sample_views.shape[2]
    aug, mask = drop_views(sample_views, prob=0.3, min_views=2)
    assert aug.shape == sample_views.shape
    assert mask.shape == (B, V)
    assert mask.dtype == torch.float32


def test_drop_views_keeps_min_views(sample_views):
    B = sample_views.shape[0]
    _, mask = drop_views(sample_views, prob=0.99, min_views=2)
    assert mask.sum(dim=1).ge(2).all()


def test_drop_views_zeros_confidence(sample_views):
    B, T, V, J, _ = sample_views.shape
    aug, mask = drop_views(sample_views, prob=0.5, min_views=2)
    for b in range(B):
        for v in range(V):
            if mask[b, v].item() == 0.0:
                assert aug[b, :, v, :, 2].abs().max().item() < 1e-6


def test_drop_views_reproducible_with_seed(sample_views):
    aug1, mask1 = drop_views(sample_views, prob=0.5, seed=42)
    aug2, mask2 = drop_views(sample_views, prob=0.5, seed=42)
    assert torch.equal(aug1, aug2)
    assert torch.equal(mask1, mask2)


def test_drop_views_prob_zero_keeps_all(sample_views):
    _, mask = drop_views(sample_views, prob=0.0, min_views=2)
    assert torch.equal(mask, torch.ones_like(mask))


def test_view_dropout_augmentation_v46_wrapper(sample_views):
    B = sample_views.shape[0]
    V = sample_views.shape[2]
    wrapper = ViewDropoutAugmentationV46(
        dropout_rate=0.5, min_views=2, curriculum=True, seed=123
    )
    aug, mask = wrapper(sample_views, progress=0.5)
    assert aug.shape == sample_views.shape
    assert mask.shape == (B, V)
    assert mask.sum(dim=1).ge(2).all()


def test_view_dropout_augmentation_v46_curriculum_progress_zero(sample_views):
    wrapper = ViewDropoutAugmentationV46(
        dropout_rate=0.5, min_views=2, curriculum=True, seed=123
    )
    _, mask = wrapper(sample_views, progress=0.0)
    assert torch.equal(mask, torch.ones_like(mask))


def test_invalid_prob():
    with pytest.raises(ValueError, match="prob must be in"):
        drop_views(torch.rand(2, 4, 17, 3), prob=-0.1, min_views=2)


def test_min_views_too_large(sample_views):
    V = sample_views.shape[2]
    with pytest.raises(ValueError, match="cannot exceed"):
        drop_views(sample_views, prob=0.2, min_views=V + 1)
