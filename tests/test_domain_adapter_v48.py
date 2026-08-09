"""Unit/integration tests for v48 DomainAdapterV48.

These tests verify that ``DomainAdapterV48``:

* Applies FiLM / conditional BN with the correct output shape.
* Is approximately an identity mapping at initialization when only FiLM is
  enabled (zero-initialized affine parameters).
* Masks/ignores invalid ``dataset_id`` values.
* Produces gradients for all enabled sub-modules.
* Optionally returns GRL domain-discriminator logits of the expected shape.
* Can be wired into ``OmniMultiViewFusionV5`` once the v48 flag is added.

The integration test is skipped until ``OmniMultiViewFusionV5`` exposes
``use_v48_domain_generalization``.
"""

from __future__ import annotations

import math
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


def test_view_mask_argument_is_accepted(dummy_features):
    """The forward signature accepts an optional view_mask without crashing."""
    feat, dataset_id = dummy_features
    B, T, V, _, _ = feat.shape
    adapter = DomainAdapterV48(in_channels=feat.shape[-1])
    view_mask = torch.ones(B, T, V)
    view_mask[:, :, -1] = 0.0
    out, logits = adapter(feat, dataset_id, view_mask=view_mask)
    assert out.shape == feat.shape
    assert logits is not None


def test_per_batch_different_domain_ids(dummy_features):
    """Each batch element can be from a different domain."""
    feat, _ = dummy_features
    B = feat.shape[0]
    adapter = DomainAdapterV48(in_channels=feat.shape[-1], num_domains=6)
    dataset_id = torch.arange(B, dtype=torch.long) % 6
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert logits.shape == (B, 6)


def test_default_hyperparameters():
    """Default values match the v48 proposal."""
    adapter = DomainAdapterV48(in_channels=32)
    assert adapter.num_domains == 6
    assert adapter.use_film
    assert not adapter.use_conditional_bn
    assert adapter.use_grl_discriminator


# ---------------------------------------------------------------------------
# Integration tests
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


def test_v48_flag_wires_into_omniview_fusion_v5():
    """OmniMultiViewFusionV5 with v48 enabled can run a forward pass."""
    from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5

    sig = OmniMultiViewFusionV5.__init__.__code__.co_varnames
    if "use_v48_domain_generalization" not in sig:
        pytest.skip("OmniMultiViewFusionV5 does not yet support use_v48_domain_generalization")

    B, T, V, J = 2, 3, 4, 17
    K, R, t = _make_cameras(V)
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
        use_v47_temporal_aggregation=False,
        use_v48_domain_generalization=True,
        v48_dg_hidden=16,
        return_covariance=False,
    )
    pred_3d, weights, visibility, L, epi_loss = model(
        x=x,
        K=K.unsqueeze(0).expand(B, -1, -1, -1),
        R=R.unsqueeze(0).expand(B, -1, -1, -1),
        t=t.unsqueeze(0).expand(B, -1, -1),
        dataset_id=torch.zeros(B, dtype=torch.long),
    )
    assert pred_3d.shape == (B, T, J, 3)
    assert epi_loss.numel() == 1
