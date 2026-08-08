import math

import torch
import pytest

from motionflow_mv.fusion.test_time_self_evolution_v27 import (
    TestTimeSelfEvolutionV27,
    build_projection_matrix,
    compute_reprojection_residual,
    triangulate_dlt_per_joint,
)


def _make_cameras(n_views: int = 4, b: int = 2, t: int = 1):
    """Return synthetic calibrated cameras looking at the origin."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = torch.tensor([3.0 * math.cos(theta), 3.0 * math.sin(theta), 1.0])
        forward = -c / c.norm()
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        up = torch.linalg.cross(right, forward)
        R = torch.stack([right, up, -forward], dim=0)
        tvec = -R @ c
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        Ks.append(K)
        Rs.append(R)
        ts.append(tvec)
    K = torch.stack(Ks).float().unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1, -1)
    R = torch.stack(Rs).float().unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1, -1)
    t = torch.stack(ts).float().unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
    return K, R, t


def _project_points(X, K, R, t):
    """Project (B,T,J,3) to (B,T,V,J,2) using calibrated cameras."""
    # X_cam = X @ R^T + t
    X_cam = torch.einsum("btvac,btjc->btvja", R, X) + t[..., None, :]  # (B,T,V,J,3)
    # image = X_cam @ K^T
    image = torch.einsum("btvac,btvjc->btvja", K, X_cam)  # (B,T,V,J,3)
    points_2d = image[..., :2] / image[..., 2:3]
    return points_2d


@pytest.fixture
def simple_scene():
    b, t, v, j = 2, 1, 4, 17
    K, R, t_cam = _make_cameras(v, b, t)
    X = torch.randn(b, t, j, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = _project_points(X, K, R, t_cam)
    return points_2d, K, R, t_cam, X


def test_build_projection_matrix():
    b, t, v = 2, 3, 4
    K = torch.randn(b, t, v, 3, 3)
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3)
    t_vec = torch.randn(b, t, v, 3)
    P = build_projection_matrix(K, R, t_vec)
    assert P.shape == (b, t, v, 3, 4)
    assert torch.allclose(P[..., :3], K @ R)
    assert torch.allclose(P[..., 3], (K @ t_vec.unsqueeze(-1)).squeeze(-1))


def test_compute_reprojection_residual_zero_for_perfect_projection(simple_scene):
    points_2d, K, R, t, X = simple_scene
    residual = compute_reprojection_residual(X, points_2d, K, R, t)
    assert residual.shape == points_2d.shape[:-1]
    assert residual.abs().max() < 1e-3


def test_triangulate_dlt_per_joint_recovers_3d(simple_scene):
    points_2d, K, R, t, X = simple_scene
    weights = torch.ones(*points_2d.shape[:-1])
    P = build_projection_matrix(K, R, t)
    X_hat = triangulate_dlt_per_joint(points_2d, P, weights)
    assert X_hat.shape == X.shape
    assert (X_hat - X).norm(dim=-1).mean() < 1e-2


def test_test_time_self_evolution_identity_with_n_iters_zero(simple_scene):
    points_2d, K, R, t, X = simple_scene
    module = TestTimeSelfEvolutionV27(n_iters=0)
    out = module(X, points_2d, K, R, t)
    assert torch.allclose(out, X)


def test_test_time_self_evolution_reduces_outlier_bias(simple_scene):
    points_2d, K, R, t, X = simple_scene
    # Corrupt one view for one joint.
    points_2d_noisy = points_2d.clone()
    points_2d_noisy[:, :, 0, 0, :] += 50.0

    module = TestTimeSelfEvolutionV27(n_iters=3, sigma_reproj=5.0)
    out = module(X, points_2d_noisy, K, R, t)
    # The refined pose should be closer to the true X than the simple
    # unweighted DLT estimate from the noisy points.
    P = build_projection_matrix(K, R, t)
    X_naive = triangulate_dlt_per_joint(points_2d_noisy, P, torch.ones(*points_2d.shape[:-1]))
    err_refined = (out - X).norm(dim=-1).mean().item()
    err_naive = (X_naive - X).norm(dim=-1).mean().item()
    assert err_refined < err_naive


def test_view_mask_zeros_out_masked_views(simple_scene):
    points_2d, K, R, t, X = simple_scene
    view_mask = torch.ones(points_2d.shape[0], points_2d.shape[1], points_2d.shape[2], dtype=torch.bool)
    view_mask[:, :, 0] = False
    module = TestTimeSelfEvolutionV27(n_iters=2)
    out = module(X, points_2d, K, R, t, view_mask=view_mask)
    assert out.shape == X.shape
