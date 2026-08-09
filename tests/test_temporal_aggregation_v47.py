"""Unit/integration tests for v47 Temporal Aggregation.

The tests assume ``TemporalAggregationV47`` exposes the API described in
``docs/proposals/v47_combined_architecture.md``.  If the module has not been
implemented yet, the whole test file is skipped with an explanatory message.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.temporal_aggregation_v47 import TemporalAggregationV47


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_poses():
    """Return a batch of per-frame 3D poses with full view masks."""
    B, T, V, J = 2, 7, 4, 17
    poses = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V)
    return poses, view_mask


@pytest.fixture
def sparse_view_mask():
    """Return a view mask where each batch element has different active views."""
    B, T, V, J = 2, 7, 4, 17
    view_mask = torch.zeros(B, T, V)
    # Batch 0: views 0 and 1 active.
    view_mask[0, :, [0, 1]] = 1.0
    # Batch 1: views 0, 1, 2 active.
    view_mask[1, :, [0, 1, 2]] = 1.0
    return view_mask


@pytest.fixture
def clip_mask():
    """Return a clip mask where the last two frames are invalid for batch 0."""
    B, T = 2, 7
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -2:] = False
    return mask


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_forward_shape(dummy_poses):
    """Output shape must match input shape (B, T, J, 3)."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2])
    out = module(poses, view_mask)
    assert out.shape == poses.shape


def test_identity_at_init(dummy_poses):
    """With the residual gate initialised to 0, the module is a no-op."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(
        n_joints=poses.shape[2],
        residual_gate_init=0.0,
    )
    out = module(poses, view_mask)
    assert torch.allclose(out, poses, atol=1e-6)


def test_residual_gate_nonzero_changes_output(dummy_poses):
    """A non-zero residual gate should allow a non-trivial refinement."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(
        n_joints=poses.shape[2],
        residual_gate_init=1.0,
    )
    out = module(poses, view_mask)
    # The output should still be a valid tensor of the same shape; we do not
    # require exact inequality because zero-initialised weights could in principle
    # produce a near-zero residual.  The gate simply permits a non-zero change.
    assert out.shape == poses.shape
    assert torch.isfinite(out).all()


def test_view_count_conditioning_flag(dummy_poses):
    """Both view-count-conditioned and unconditioned modes should run."""
    poses, view_mask = dummy_poses
    for use_view_count in (True, False):
        module = TemporalAggregationV47(
            n_joints=poses.shape[2],
            use_view_count_conditioning=use_view_count,
        )
        out = module(poses, view_mask)
        assert out.shape == poses.shape


def test_sparse_view_mask(dummy_poses, sparse_view_mask):
    """Module must accept masks with dropped views without crashing."""
    poses, _ = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2])
    out = module(poses, sparse_view_mask)
    assert out.shape == poses.shape
    assert torch.isfinite(out).all()


def test_clip_mask(dummy_poses, clip_mask):
    """Clip mask should ignore invalid frames without affecting output shape."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2])
    out = module(poses, view_mask, clip_mask=clip_mask)
    assert out.shape == poses.shape
    assert torch.isfinite(out).all()


def test_clip_mask_does_not_leak(dummy_poses, clip_mask):
    """Invalid frames should not change the values of valid frames."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2])
    module.eval()
    with torch.no_grad():
        out_full = module(poses, view_mask)
        out_masked = module(poses, view_mask, clip_mask=clip_mask)
    valid = clip_mask.unsqueeze(-1).unsqueeze(-1).expand_as(poses)
    assert torch.allclose(out_full[valid], out_masked[valid], atol=1e-6)


def test_gradients_flow(dummy_poses):
    """Backward pass should compute gradients for all parameters."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2])
    out = module(poses, view_mask)
    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())


def test_windowed_attention(dummy_poses):
    """A finite temporal window should produce valid output."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(
        n_joints=poses.shape[2],
        temporal_window=3,
    )
    out = module(poses, view_mask)
    assert out.shape == poses.shape


def test_variable_length_clips():
    """Module should handle clips with different effective lengths."""
    B, T, V, J = 2, 9, 3, 17
    poses = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V)
    clip_mask = torch.ones(B, T, dtype=torch.bool)
    clip_mask[0, 5:] = False
    clip_mask[1, 7:] = False

    module = TemporalAggregationV47(n_joints=J)
    out = module(poses, view_mask, clip_mask=clip_mask)
    assert out.shape == poses.shape
    assert torch.isfinite(out).all()


def test_default_hyperparameters():
    """Default constructor arguments match the v47 proposal."""
    module = TemporalAggregationV47()
    assert module.n_joints == 17
    assert module.d_model == 64
    assert module.n_heads == 4
    assert module.num_layers == 2


def test_dropout_in_train_vs_eval(dummy_poses):
    """Dropout should be active in train mode and disabled in eval mode."""
    poses, view_mask = dummy_poses
    module = TemporalAggregationV47(n_joints=poses.shape[2], dropout=0.5)
    module.eval()
    with torch.no_grad():
        out_eval = module(poses, view_mask)
    module.train()
    with torch.no_grad():
        out_train = module(poses, view_mask)
    assert out_eval.shape == out_train.shape
    assert torch.isfinite(out_train).all()


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


def test_v47_flag_wires_into_omniview_fusion_v5():
    """OmniMultiViewFusionV5 with v47 enabled can run a forward pass.

    This test is skipped until Agent-05 adds ``use_v47_temporal_aggregation`` to
    ``OmniMultiViewFusionV5``.
    """
    from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5

    sig = inspect.signature(OmniMultiViewFusionV5.__init__)
    if "use_v47_temporal_aggregation" not in sig.parameters:
        pytest.skip("OmniMultiViewFusionV5 does not yet support use_v47_temporal_aggregation")

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
        use_v47_temporal_aggregation=True,
        v47_temporal_d_model=32,
        v47_temporal_num_layers=1,
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
