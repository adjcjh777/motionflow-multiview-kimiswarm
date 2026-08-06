"""CPU smoke tests for confidence-aware per-joint view dropout."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from motionflow_mv.data.confidence_resample_dropout import (
    ConfidenceResampleDropout,
    confidence_resample_view_dropout,
)


def test_basic_shape_and_independence():
    B, T, V, J, C = 2, 5, 4, 17, 3
    x = torch.rand(B, T, V, J, C)
    x[..., 2] = torch.rand(B, T, V, J)

    y = confidence_resample_view_dropout(x, dropout_rate=0.3)
    assert y.shape == x.shape

    y0 = confidence_resample_view_dropout(x, dropout_rate=0.0)
    assert torch.equal(y0, x)

    y1 = confidence_resample_view_dropout(x, dropout_rate=0.5, generator=torch.Generator().manual_seed(42))
    y2 = confidence_resample_view_dropout(x, dropout_rate=0.5, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(y1, y2)


def test_confidence_bias_preserves_high_confidence():
    V, J, C = 4, 5, 3
    x = torch.zeros(1, V, J, C)
    x[0, 0, :, 2] = 100.0
    x[0, 1:, :, 2] = 0.01

    kept = 0
    n = 100
    for _ in range(n):
        out = confidence_resample_view_dropout(x, dropout_rate=0.5, resample=False)
        if out[0, 0, 0, 2].item() > 50.0:
            kept += 1
    assert kept >= int(0.95 * n)


def test_min_views_enforced():
    V, J, C = 4, 5, 3
    x = torch.rand(1, V, J, C)
    x[..., 2] = torch.rand(V, J)
    y = confidence_resample_view_dropout(x, dropout_rate=0.9, resample=False, min_views=2)
    for j in range(J):
        assert y[0, :, j, 2].gt(0).sum().item() >= 2


def test_resample_preserves_shape_and_kept_count():
    V, J, C = 4, 5, 3
    x = torch.rand(1, V, J, C)
    x[..., 2] = torch.rand(V, J)
    y = confidence_resample_view_dropout(x, dropout_rate=0.5, resample=True, min_views=1)
    assert y.shape == x.shape
    nonzero = (y[..., 2] > 0).float().mean().item()
    assert nonzero > 0.99


def test_augmenter_wrapper_state():
    x = torch.rand(2, 4, 17, 3)
    x[..., 2] = torch.rand(2, 4, 17)
    aug = ConfidenceResampleDropout(dropout_rate=0.2, resample=True, min_views=2, seed=123)
    y = aug(x)
    assert y.shape == x.shape

    state = aug.state_dict()
    aug2 = ConfidenceResampleDropout()
    aug2.load_state_dict(state)
    assert aug2.dropout_rate == 0.2
    assert aug2.resample is True
    assert aug2.min_views == 2


def test_different_confidence_channel():
    x = torch.rand(1, 4, 5, 3)
    x[..., 0] = torch.rand(1, 4, 5)
    y = confidence_resample_view_dropout(x, dropout_rate=0.3, confidence_channel=0)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_basic_shape_and_independence()
    test_confidence_bias_preserves_high_confidence()
    test_min_views_enforced()
    test_resample_preserves_shape_and_kept_count()
    test_augmenter_wrapper_state()
    test_different_confidence_channel()
    print("All confidence_resample_dropout tests passed.")
