"""Unit tests for DLT triangulation."""

import numpy as np
import pytest

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import triangulate_dlt


def _random_camera(rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(0)
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 800.0
    K[:2, 2] = rng.uniform(300, 340, size=2)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = rng.standard_normal(3) * 2.0
    return Camera(K=K, R=R, t=t)


def test_triangulate_dlt_known_point():
    """Project a known 3D point into 4 views and recover it."""
    X_world = np.array([1.0, 0.5, 3.0])
    rng = np.random.default_rng(42)
    cameras = [_random_camera(rng) for _ in range(4)]

    points_2d = []
    proj_matrices = []
    for cam in cameras:
        P = cam.projection_matrix
        x_h = P @ np.append(X_world, 1.0)
        x = x_h[:2] / x_h[2]
        points_2d.append(x)
        proj_matrices.append(P)

    X_recovered = triangulate_dlt(np.array(points_2d), np.array(proj_matrices))
    np.testing.assert_allclose(X_recovered, X_world, atol=1e-4)


def test_triangulate_dlt_batched_lstsq_matches_single_joint():
    """Batched lstsq DLT should match the single-joint torch DLT."""
    import torch
    from motionflow_mv.fusion.triangulation import (
        triangulate_dlt_batched_lstsq,
        triangulate_dlt_torch,
    )

    rng = np.random.default_rng(42)
    N, V, J = 4, 4, 17

    proj_matrices = torch.randn(N, V, 3, 4, dtype=torch.float64)
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64) * 100
    weights = torch.rand(N, V, J, dtype=torch.float64)

    pred_batched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)

    pred_single = torch.zeros(N, J, 3, dtype=torch.float64)
    for j in range(J):
        pred_single[:, j, :] = triangulate_dlt_torch(
            points_2d[:, :, j, :], proj_matrices, weights[:, :, j]
        )

    torch.testing.assert_close(pred_batched, pred_single, atol=1e-5, rtol=1e-5)


def test_bayesian_tri_v2_forward_backward():
    """BayesianTriV2 model should run forward and backward without error."""
    import torch
    from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
        RayAttentionFusionModelBayesianTriV2,
    )

    def _make_cameras(n_views: int = 4):
        cameras = []
        for i in range(n_views):
            theta = 2 * np.pi * i / n_views
            c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
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
            cameras.append(Camera(K=K, R=R, t=t))
        return cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV2(
        j=J, d=64, n_views=V, gn_iters=2, epipolar_loss_weight=0.05
    )
    pred, weights, pp_delta, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert epi_loss.shape == ()
    loss = pred.mean() + 0.0 * epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


if __name__ == "__main__":
    test_triangulate_dlt_known_point()
    test_triangulate_dlt_batched_lstsq_matches_single_joint()
    test_bayesian_tri_v2_forward_backward()
    print("triangulation tests passed")
