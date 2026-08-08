"""Unit tests for motionflow_mv/fusion/multiview_geometry_fusion_v25.py.

Covers the public contract of ``MultiViewGeometryFusionV25``:
- forward shape ``(B, T, V, J, 3) -> (B, T, J, 3)``
- identity-at-init behaviour
- gradient flow through inputs
- view masking
- shape compatibility for J=17 and J=28
"""

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    MultiViewGeometryFusionV25,
    build_projection_matrix,
    compute_rays,
    ray_intersection_logit,
    triangulate_initial,
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
    T: int = 3,
    V: int = 4,
    J: int = 17,
) -> tuple:
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


# -----------------------------------------------------------------------------
# Core shape / forward tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("J", [17, 28])
def test_forward_shape(J):
    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch(B=2, T=3, V=4, J=J)
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 3, J, 3)


# -----------------------------------------------------------------------------
# Identity at init
# -----------------------------------------------------------------------------

def test_identity_at_init():
    module = MultiViewGeometryFusionV25(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert torch.allclose(out, pred_3d_init, atol=1e-5)


def test_identity_at_init_without_pred():
    module = MultiViewGeometryFusionV25(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=False,
        use_learned_depth_triangulation=False,
    )
    points_2d, K, R, t, _, _ = _make_batch()
    out, _ = module(points_2d, K, R, t)
    expected = triangulate_initial(points_2d[..., :2], K, R, t)
    assert torch.allclose(out, expected, atol=1e-5)


# -----------------------------------------------------------------------------
# Gradient flow
# -----------------------------------------------------------------------------

def test_gradient_flow():
    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=4)
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


# -----------------------------------------------------------------------------
# View masking
# -----------------------------------------------------------------------------

def test_view_mask_ignores_dropped_view():
    module = MultiViewGeometryFusionV25(d=64, n_heads=2, n_views=4)
    points_2d, K, R, t, _, _ = _make_batch()
    view_mask = torch.ones(2, 3, 4).bool()
    view_mask[:, :, -1] = False

    out, _ = module(points_2d, K, R, t, view_mask=view_mask)
    assert out.shape == (2, 3, 17, 3)
    assert out.isfinite().all()


# -----------------------------------------------------------------------------
# Toggle coverage
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("use_geom_attn", [True, False])
@pytest.mark.parametrize("use_learned_depth", [True, False])
def test_toggles_forward(use_geom_attn: bool, use_learned_depth: bool):
    module = MultiViewGeometryFusionV25(
        d=64,
        n_heads=2,
        n_views=4,
        use_geometry_attention=use_geom_attn,
        use_learned_depth_triangulation=use_learned_depth,
    )
    points_2d, K, R, t, pred_3d_init, view_mask = _make_batch()
    out, _ = module(points_2d, K, R, t, pred_3d_init=pred_3d_init, view_mask=view_mask)
    assert out.shape == (2, 3, 17, 3)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def test_build_projection_matrix_shape():
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(2, 3, 4, 3, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(2, 3, 4, 3, 3)
    t = torch.zeros(2, 3, 4, 3)
    P = build_projection_matrix(K, R, t)
    assert P.shape == (2, 3, 4, 3, 4)


def test_triangulate_initial_shape():
    points_2d, K, R, t, _, _ = _make_batch()
    X = triangulate_initial(points_2d[..., :2], K, R, t)
    assert X.shape == (2, 3, 17, 3)


def test_compute_rays_shape():
    points_2d, K, R, t, _, _ = _make_batch()
    centre, direction = compute_rays(points_2d[..., :2], K, R, t)
    assert centre.shape == (2, 3, 4, 3)
    assert direction.shape == (2, 3, 4, 17, 3)
    assert torch.allclose(
        direction.norm(dim=-1),
        torch.ones_like(direction.norm(dim=-1)),
        atol=1e-5,
    )


def test_ray_intersection_logit_shape():
    points_2d, K, R, t, _, _ = _make_batch()
    centre, direction = compute_rays(points_2d[..., :2], K, R, t)
    sigma_d = torch.tensor(0.5)
    sigma_a = torch.tensor(0.5)
    logit = ray_intersection_logit(centre, direction, sigma_d, sigma_a)
    assert logit.shape == (2, 3, 4, 4, 17)


if __name__ == "__main__":
    test_forward_shape(17)
    test_forward_shape(28)
    test_identity_at_init()
    test_identity_at_init_without_pred()
    test_gradient_flow()
    test_view_mask_ignores_dropped_view()
    for ga in [True, False]:
        for ld in [True, False]:
            test_toggles_forward(ga, ld)
    test_build_projection_matrix_shape()
    test_triangulate_initial_shape()
    test_compute_rays_shape()
    test_ray_intersection_logit_shape()
    print("All MultiViewGeometryFusionV25 unit tests passed")
