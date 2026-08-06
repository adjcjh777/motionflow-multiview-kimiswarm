"""Smoke tests for the multi-view sync augmentation module and wrapper."""

import torch
import torch.nn as nn

from motionflow_mv.data.multiview_temporal_jitter import MultiViewSyncAugmentation
from motionflow_mv.models.data_augmentation_multiview_wrapper import (
    MultiViewDataAugmentationWrapper,
)


class DummyModel(nn.Module):
    """Dummy model that returns the first two channels of x and a constant."""

    def __init__(self):
        super().__init__()
        self.register_buffer("dummy", torch.ones(1))

    def forward(self, x, **kwargs):
        return x[..., :2], torch.ones(x.shape[0])


def test_multiview_sync_augmentation_preserves_shape():
    aug = MultiViewSyncAugmentation()
    x = torch.randn(2, 13, 4, 17, 3)
    x_aug = aug(x)
    assert x_aug.shape == x.shape


def test_multiview_sync_augmentation_changes_training_mode_only():
    aug = MultiViewSyncAugmentation(subclip_len=5, translation_std=5.0, noise_std=1.0)
    x = torch.randn(2, 10, 4, 17, 3)
    aug.train()
    x_aug = aug(x)
    aug.eval()
    x_eval = aug(x)
    assert not torch.equal(x_aug, x)
    assert torch.equal(x_eval, x)


def test_multiview_sync_augmentation_view_dropout():
    aug = MultiViewSyncAugmentation(view_dropout_rate=0.5, min_views=2)
    x = torch.ones(4, 10, 4, 17, 3)
    x[..., 2] = 1.0  # confidences all 1
    aug.train()
    x_aug = aug(x)
    # At least 2 views should be kept per sample.
    nonzero = (x_aug[..., 2] > 0).float().sum(dim=2)  # (B, T, J)
    assert (nonzero > 0).all()


def test_wrapper_forwards_and_delegates_state_dict():
    base = DummyModel()
    wrapped = MultiViewDataAugmentationWrapper(base)
    x = torch.randn(2, 5, 4, 17, 3)
    wrapped.train()
    out_train, _ = wrapped(x)
    assert out_train.shape == (2, 5, 4, 17, 2)

    wrapped.eval()
    out_eval, _ = wrapped(x)
    assert torch.equal(out_eval, x[..., :2])

    # State dict should match the base model.
    assert list(wrapped.state_dict().keys()) == list(base.state_dict().keys())
