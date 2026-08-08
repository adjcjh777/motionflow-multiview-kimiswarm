"""Unit tests for motionflow_mv/fusion/temporal_geometry_fusion_v26.py.

Covers the public contract of ``TemporalGeometryFusionV26``:
- forward shape ``(B, T, V, J, 3) -> (B, T, J, 3)``
- identity-at-init behaviour (falls back to v25)
- gradient flow through inputs
- view masking
- temporal boundary handling (T < temporal_window)
- toggle coverage
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from motionflow_mv.fusion.multiview_geometry_fusion_v25 import triangulate_initial
from motionflow_mv.fusion.temporal_geometry_fusion_v26 import (
    TemporalGeometryAttention,
    TemporalGeometryFusionV26,
)


def _make_cameras(n_views: int = 4) -> tuple:
    """Build a simple circular camera rig."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    return (
        torch.from_numpy(np.stack(Ks)).float(),
        torch.from_numpy(np.stack(Rs)).float(),
        torch.from_numpy(np.stack(ts)).float(),
    )


def _project_points(
    joints_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project world points into all views; returns (F, V, J, 2)."""
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, joints_3d) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


def _make_batch(
    B: int = 2,
    T: int = 5,
    V: int = 4,
    J: int = 17,
):
    """Return synthetic (points_2d, K, R, t, pred_3d_init, view_mask)."""
    K, R, t = _make_cameras(V)
    torch.manual_seed(42)
    joints_3d = torch.randn(T, J, 3) * 0.3
    points_2d = _project_points(joints_3d, K, R, t)

    K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    t = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    points_2d = points_2d.unsqueeze(0).expand(B, -1, -1, -1, -1)
    confidence = torch.ones(B, T, V, J)
    points_2d = torch.cat([points_2d, confidence[..., None]], dim=-1)

    pred_3d_init = joints_3d.unsqueeze(0).expand(B, -1, -1, -1)
    view_mask = torch.ones(B, T, V).bool()
    return points_2d, K, R, t, pred_3d_init, view_mask


# ---------------------------------------------------------------------------
# Core shape / forward tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("J", [17, 28])
def test_forward_shape(J):
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4, temporal_window=3)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch(B=2, T=5, V=4, J=J)
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 5, J, 3)


def test_temporal_attention_forward_shape():
    B, T, V, J, d = 2, 5, 4, 17, 64
    tokens = torch.randn(B, T, V, J, d)
    epipolar_dist = torch.rand(B, T, V, V, J)
    ray_logit = torch.randn(B, T, V, V, J)
    attn = TemporalGeometryAttention(d=d, n_heads=2, n_views=V, temporal_window=3)
    out = attn(tokens, epipolar_dist, ray_logit)
    assert out.shape == (B, T, V, J, d)


# ---------------------------------------------------------------------------
# Identity at init
# ---------------------------------------------------------------------------

def test_identity_at_init():
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)


def test_identity_at_init_with_temporal_attention_enabled():
    """With residual gate initialised to 0, temporal attention is a no-op at init."""
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=True,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)
    # Verify the gate was initialised to zero.
    for layer in module.temporal_attn_layers:
        assert layer.residual_gate.item() == pytest.approx(0.0, abs=1e-6)


def test_identity_at_init_without_pred():
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_temporal_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, _, _ = _make_batch()
    out, _ = module(points_2d, K, R, t)
    expected = triangulate_initial(points_2d[..., :2], K, R, t)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    points_2d.requires_grad_(True)
    K.requires_grad_(True)
    R.requires_grad_(True)
    t.requires_grad_(True)
    pred_3d_init.requires_grad_(True)

    out, geom_loss = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    loss = out.mean() + geom_loss
    loss.backward()

    assert points_2d.grad is not None
    assert K.grad is not None
    assert R.grad is not None
    assert t.grad is not None
    assert pred_3d_init.grad is not None


def test_residual_gate_learnable():
    """The residual gate should scale the temporal attention output and receive gradients."""
    B, T, V, J, d = 2, 5, 4, 17, 64
    tokens = torch.randn(B, T, V, J, d, requires_grad=True)
    epipolar_dist = torch.rand(B, T, V, V, J)
    ray_logit = torch.randn(B, T, V, V, J)
    attn = TemporalGeometryAttention(d=d, n_heads=2, n_views=V, temporal_window=3)

    # Default gate is 0, so the output is zero.
    out_zero = attn(tokens, epipolar_dist, ray_logit)
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6)

    # Open the gate and randomise out_proj; the residual becomes non-zero.
    nn.init.constant_(attn.residual_gate, 1.0)
    torch.manual_seed(123)
    nn.init.xavier_uniform_(attn.out_proj.weight)
    out_open = attn(tokens, epipolar_dist, ray_logit)
    assert not torch.allclose(out_open, torch.zeros_like(out_open), atol=1e-6)

    loss = out_open.mean()
    loss.backward()
    assert attn.residual_gate.grad is not None


# ---------------------------------------------------------------------------
# View masking
# ---------------------------------------------------------------------------

def test_view_mask_ignores_dropped_view():
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, _, _ = _make_batch()
    view_mask = torch.ones(2, 5, 4).bool()
    view_mask[:, :, -1] = False

    out, _ = module(points_2d, K, R, t, view_mask=view_mask)
    assert out.shape == (2, 5, 17, 3)
    assert out.isfinite().all()


# ---------------------------------------------------------------------------
# Temporal boundary handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 2, 3, 7])
def test_temporal_window_larger_than_clip(T):
    """Clip length shorter than temporal window should not crash."""
    module = TemporalGeometryFusionV26(d=64, n_heads=2, n_views=4, temporal_window=5)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch(B=2, T=T, V=4)
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, T, 17, 3)


# ---------------------------------------------------------------------------
# Toggle coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_geom_attn", [True, False])
@pytest.mark.parametrize("use_temporal", [True, False])
@pytest.mark.parametrize("use_learned_depth", [True, False])
def test_toggles_forward(use_geom_attn: bool, use_temporal: bool, use_learned_depth: bool):
    module = TemporalGeometryFusionV26(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=use_geom_attn,
        use_temporal_geometry_attention=use_temporal,
        use_learned_depth_triangulation=use_learned_depth,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 5, 17, 3)


def test_invalid_head_dimension_raises():
    with pytest.raises(ValueError):
        TemporalGeometryAttention(d=64, n_heads=3, n_views=4)


def test_invalid_temporal_window_raises():
    with pytest.raises(ValueError):
        TemporalGeometryAttention(d=64, n_heads=4, n_views=4, temporal_window=4)


if __name__ == "__main__":
    test_forward_shape(17)
    test_forward_shape(28)
    test_temporal_attention_forward_shape()
    test_identity_at_init()
    test_identity_at_init_with_temporal_attention_enabled()
    test_identity_at_init_without_pred()
    test_gradient_flow()
    test_residual_gate_learnable()
    test_view_mask_ignores_dropped_view()
    test_temporal_window_larger_than_clip(1)
    test_temporal_window_larger_than_clip(2)
    test_temporal_window_larger_than_clip(3)
    test_temporal_window_larger_than_clip(7)
    for ga in [True, False]:
        for ut in [True, False]:
            for ud in [True, False]:
                test_toggles_forward(ga, ut, ud)
    test_invalid_head_dimension_raises()
    test_invalid_temporal_window_raises()
    print("All TemporalGeometryFusionV26 unit tests passed")
