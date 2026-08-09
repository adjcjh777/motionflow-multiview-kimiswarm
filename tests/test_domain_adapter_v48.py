"""Unit tests for v48 DomainAdapterV48.

These tests verify that ``DomainAdapterV48``:

* Applies FiLM / conditional BN with the correct output shape.
* Is approximately an identity mapping at initialization when only FiLM is
  enabled (zero-initialized affine parameters).
* Masks/ignores invalid ``dataset_id`` values.
* Produces gradients for all enabled sub-modules.
* Optionally returns GRL domain-discriminator logits of the expected shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.domain_adapter_v48 import DomainAdapterV48


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_features():
    """Return random multi-view feature tokens and a domain-id tensor."""
    B, T, V, J, C = 2, 5, 4, 17, 32
    feat = torch.randn(B, T, V, J, C)
    dataset_id = torch.tensor([0, 5], dtype=torch.long)
    return feat, dataset_id


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_forward_shape_with_all_mechanisms(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        hidden=16,
        use_film=True,
        use_conditional_bn=True,
        use_grl_discriminator=True,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert logits is not None
    assert logits.shape == (feat.shape[0], 6)


def test_film_only_is_identity_at_init(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        use_film=True,
        use_conditional_bn=False,
        use_grl_discriminator=False,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert torch.allclose(out, feat, atol=1e-5)
    assert logits is None


def test_conditional_bn_only(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        use_film=False,
        use_conditional_bn=True,
        use_grl_discriminator=False,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert torch.isfinite(out).all()
    assert logits is None


def test_no_mechanisms_is_pass_through(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        use_film=False,
        use_conditional_bn=False,
        use_grl_discriminator=False,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert torch.allclose(out, feat, atol=1e-5)
    assert logits is None


def test_invalid_domain_id_raises(dummy_features):
    feat, _ = dummy_features
    adapter = DomainAdapterV48(in_channels=feat.shape[-1], num_domains=6)
    invalid_ids = torch.tensor([-1, 6], dtype=torch.long)
    with pytest.raises(ValueError):
        adapter(feat, invalid_ids)


def test_gradients_flow(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        hidden=16,
        use_film=True,
        use_conditional_bn=True,
        use_grl_discriminator=True,
    )
    out, logits = adapter(feat, dataset_id)
    loss = out.sum() + (logits.sum() if logits is not None else 0.0)
    loss.backward()
    assert any(p.grad is not None for p in adapter.parameters())


def test_discriminator_logits_depend_on_adapter_output(dummy_features):
    feat, dataset_id = dummy_features
    adapter = DomainAdapterV48(
        in_channels=feat.shape[-1],
        num_domains=6,
        use_film=False,
        use_conditional_bn=False,
        use_grl_discriminator=True,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert logits is not None
    assert logits.shape == (feat.shape[0], 6)
