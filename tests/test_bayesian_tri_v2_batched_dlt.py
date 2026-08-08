"""Unit tests for the batched DLT used by Bayesian tri v2.

These tests focus on ``motionflow_mv.fusion.triangulation.triangulate_dlt_batched_lstsq``,
the fully-batched least-squares DLT routine that backs
``RayAttentionFusionModelBayesianTriV2``.
"""

import numpy as np
import pytest
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.triangulation import (
    triangulate_dlt,
    triangulate_dlt_batched_lstsq,
    triangulate_dlt_torch,
)


def _make_cameras(n_views: int = 4, seed: int = 42):
    """Create a deterministic circular multi-view rig."""
    rng = np.random.default_rng(seed)
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


def _project_points(points_3d, cameras):
    """Project ``points_3d`` (B, J, 3) through a list of cameras.

    Returns:
        points_2d: (B, V, J, 2)
        proj_matrices: (V, 3, 4)
    """
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    B, J, _ = points_3d.shape
    V = len(cameras)
    points_2d = np.zeros((B, V, J, 2))
    for v in range(V):
        P = proj_matrices[v]
        X_h = np.concatenate([points_3d, np.ones((B, J, 1))], axis=-1)
        x_h = X_h @ P.T
        points_2d[:, v, :, 0] = x_h[..., 0] / x_h[..., 2]
        points_2d[:, v, :, 1] = x_h[..., 1] / x_h[..., 2]
    return points_2d, proj_matrices


def test_batched_dlt_output_shape_and_dtype():
    """Output shape and dtype should match the batched input."""
    N, V, J = 3, 4, 7
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64)
    proj_matrices = torch.randn(V, 3, 4, dtype=torch.float64)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices)
    assert pred.shape == (N, J, 3)
    assert pred.dtype == points_2d.dtype

    weights = torch.rand(N, V, J, dtype=torch.float64)
    pred_w = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    assert pred_w.shape == (N, J, 3)


def test_batched_dlt_recover_known_points():
    """Project known 3D points through real cameras and recover them."""
    rng = np.random.default_rng(123)
    cameras = _make_cameras(4)
    B, J = 2, 9
    points_3d = rng.uniform(-1.0, 1.0, (B, J, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)
    weights = torch.rand(points_2d.shape[:3], dtype=torch.float64)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    pred_np = pred.detach().numpy()

    np.testing.assert_allclose(pred_np, points_3d, atol=1e-8)


def test_batched_dlt_matches_single_joint_torch():
    """Batched result should coincide with per-joint ``triangulate_dlt_torch``."""
    N, V, J = 3, 4, 11
    rng = np.random.default_rng(7)
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64)
    proj_matrices = torch.randn(N, V, 3, 4, dtype=torch.float64)
    weights = rng.random((N, V, J))
    weights = torch.from_numpy(weights).to(torch.float64)

    pred_batched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)

    pred_single = torch.zeros(N, J, 3, dtype=torch.float64)
    for j in range(J):
        pred_single[:, j, :] = triangulate_dlt_torch(
            points_2d[:, :, j, :], proj_matrices, weights[:, :, j]
        )

    torch.testing.assert_close(pred_batched, pred_single, atol=1e-5, rtol=1e-5)


def test_batched_dlt_weight_scale_invariance():
    """Multiplying all weights by a constant should not change the result."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(99)
    points_3d = rng.uniform(-1.0, 1.0, (2, 5, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)
    weights = torch.rand(points_2d.shape[:3], dtype=torch.float64)

    pred1 = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    pred2 = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights * 10.0)

    torch.testing.assert_close(pred1, pred2, atol=1e-7, rtol=1e-7)


def test_batched_dlt_gradients_flow():
    """The batched DLT should be differentiable w.r.t. inputs and weights."""
    N, V, J = 2, 4, 6
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64, requires_grad=True)
    proj_matrices = torch.randn(N, V, 3, 4, dtype=torch.float64)
    weights = torch.rand(N, V, J, dtype=torch.float64, requires_grad=True)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    loss = pred.sum()
    loss.backward()

    assert points_2d.grad is not None
    assert weights.grad is not None
    assert torch.isfinite(points_2d.grad).all()
    assert torch.isfinite(weights.grad).all()


def test_batched_dlt_matches_naive_numpy_dlt():
    """Batched result should match the reference numpy DLT for a single point."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(2027)
    X = rng.uniform(-1.0, 1.0, 3)

    points_2d_list = []
    proj_matrices_list = []
    for cam in cameras:
        P = cam.projection_matrix
        x_h = P @ np.append(X, 1.0)
        points_2d_list.append(x_h[:2] / x_h[2])
        proj_matrices_list.append(P)

    pred_np = triangulate_dlt(
        np.array(points_2d_list), np.array(proj_matrices_list)
    )

    points_2d = torch.from_numpy(np.array(points_2d_list)).to(torch.float64)
    proj_matrices = torch.from_numpy(np.array(proj_matrices_list)).to(torch.float64)
    # (V, 2) -> (N=1, V, J=1, 2)
    points_2d = points_2d.unsqueeze(0).unsqueeze(2).contiguous()
    proj_matrices = proj_matrices.unsqueeze(0).expand(1, -1, -1, -1)

    pred_batched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices)

    np.testing.assert_allclose(pred_batched[0, 0].detach().numpy(), pred_np, atol=1e-8)


def test_batched_dlt_precision_matrix_robust_to_indefinite_matrix():
    """Precision-matrix path should survive element-wise clamped/indefinite input."""
    N, V, J = 2, 4, 7
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64)
    proj_matrices = torch.randn(V, 3, 4, dtype=torch.float64)

    # Start with a valid precision matrix (diagonal, positive).
    precision = torch.eye(2, dtype=torch.float64).view(1, 1, 1, 2, 2)
    precision = precision.expand(N, V, J, -1, -1).clone()
    # Element-wise clamping can make the matrix indefinite, mimicking upstream code.
    precision = precision.clamp(min=-1e3, max=1e3)

    pred = triangulate_dlt_batched_lstsq(
        points_2d, proj_matrices, precision_matrix=precision
    )
    assert pred.shape == (N, J, 3)
    assert torch.isfinite(pred).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
