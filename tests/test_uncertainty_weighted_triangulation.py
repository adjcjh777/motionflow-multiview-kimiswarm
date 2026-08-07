"""Tests for uncertainty-weighted triangulation."""

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.uncertainty_weighted_triangulation import (
    UncertaintyWeightedTriangulation,
    triangulate_uncertainty_weighted,
    triangulate_uncertainty_weighted_batched,
)


def _make_cameras(n_views: int = 4, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(42)
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
        cameras.append((K, R, t))
    return cameras


def _project_point(X_world, cameras):
    points_2d = []
    proj_matrices = []
    for K, R, t in cameras:
        P = K @ np.hstack([R, t[:, None]])
        x_h = P @ np.append(X_world, 1.0)
        x = x_h[:2] / x_h[2]
        points_2d.append(x)
        proj_matrices.append(P)
    return np.stack(points_2d, axis=0), np.stack(proj_matrices, axis=0)


def test_triangulate_uncertainty_weighted_known_point():
    """Triangulate a known point with identity covariances."""
    X_true = np.array([1.0, 0.5, 3.0])
    cameras = _make_cameras(4)
    points_2d, proj_matrices = _project_point(X_true, cameras)

    points_2d = torch.from_numpy(points_2d).float()
    proj_matrices = torch.from_numpy(proj_matrices).float()
    covariances = torch.eye(2).unsqueeze(0).expand(4, -1, -1).float()

    X_recovered = triangulate_uncertainty_weighted(
        points_2d, proj_matrices, covariances=covariances
    )
    np.testing.assert_allclose(X_recovered.detach().numpy(), X_true, atol=1e-4)


def test_triangulate_uncertainty_weighted_downweights_noisy_view():
    """A high-covariance noisy view should have reduced influence."""
    X_true = np.array([1.0, 0.5, 3.0])
    cameras = _make_cameras(4)
    points_2d, proj_matrices = _project_point(X_true, cameras)

    # Corrupt the first view heavily.
    points_2d = points_2d.copy()
    points_2d[0] += np.array([50.0, -30.0])

    points_2d = torch.from_numpy(points_2d).float()
    proj_matrices = torch.from_numpy(proj_matrices).float()

    # Without uncertainty weighting, the corrupted view harms the estimate.
    X_unweighted = triangulate_uncertainty_weighted(points_2d, proj_matrices)
    err_unweighted = np.linalg.norm(X_unweighted.detach().numpy() - X_true)

    # With a large covariance on the corrupted view, the estimate should
    # approach the true point.
    covariances = torch.eye(2).unsqueeze(0).expand(4, -1, -1).clone().float()
    covariances[0] *= 1e6
    X_weighted = triangulate_uncertainty_weighted(
        points_2d, proj_matrices, covariances=covariances
    )
    err_weighted = np.linalg.norm(X_weighted.detach().numpy() - X_true)

    assert err_weighted < err_unweighted


def test_triangulate_uncertainty_weighted_is_differentiable():
    """Gradient should flow through all inputs."""
    X_true = np.array([1.0, 0.5, 3.0])
    cameras = _make_cameras(4)
    points_2d, proj_matrices = _project_point(X_true, cameras)

    points_2d = torch.from_numpy(points_2d).float().requires_grad_(True)
    proj_matrices = torch.from_numpy(proj_matrices).float().requires_grad_(True)
    covariances = torch.eye(2).unsqueeze(0).expand(4, -1, -1).float().requires_grad_(True)

    X = triangulate_uncertainty_weighted(
        points_2d, proj_matrices, covariances=covariances
    )
    loss = X.sum()
    loss.backward()

    assert points_2d.grad is not None
    assert proj_matrices.grad is not None
    assert covariances.grad is not None


def test_triangulate_uncertainty_weighted_batched():
    """Batched version should match per-joint single-joint results."""
    rng = np.random.default_rng(42)
    N, V, J = 3, 4, 5
    cameras = _make_cameras(V)

    # Generate N*J random 3D points.
    points_3d = rng.standard_normal((N, J, 3)) * 2.0
    points_2d_all = []
    for i in range(N):
        for j in range(J):
            X = points_3d[i, j]
            points_2d, _ = _project_point(X, cameras)
            points_2d_all.append(points_2d)
    points_2d_all = np.stack(points_2d_all, axis=0).reshape(N, J, V, 2)
    points_2d_all = np.transpose(points_2d_all, (0, 2, 1, 3))  # (N, V, J, 2)

    _, proj_matrices = _project_point(np.zeros(3), cameras)
    proj_matrices = torch.from_numpy(proj_matrices).float()
    points_2d_all = torch.from_numpy(points_2d_all).float()
    covariances = torch.eye(2).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    covariances = covariances.expand(N, V, J, -1, -1).float()

    X_batched = triangulate_uncertainty_weighted_batched(
        points_2d_all, proj_matrices, covariances=covariances
    )
    assert X_batched.shape == (N, J, 3)

    # Compare against per-joint single-joint triangulation.
    X_single = torch.zeros(N, J, 3)
    for i in range(N):
        for j in range(J):
            X_single[i, j] = triangulate_uncertainty_weighted(
                points_2d_all[i, :, j, :],
                proj_matrices,
                covariances=covariances[i, :, j, :, :],
            )

    torch.testing.assert_close(X_batched, X_single, atol=1e-5, rtol=1e-5)


def test_uncertainty_weighted_triangulation_module_forward_backward():
    """The learnable module should run forward and backward."""
    rng = np.random.default_rng(42)
    N, V, J = 2, 4, 17
    cameras = _make_cameras(V)

    points_3d = rng.standard_normal((N, J, 3)) * 2.0
    points_2d_all = []
    for i in range(N):
        for j in range(J):
            X = points_3d[i, j]
            points_2d, _ = _project_point(X, cameras)
            points_2d_all.append(points_2d)
    points_2d_all = np.stack(points_2d_all, axis=0).reshape(N, J, V, 2)
    points_2d_all = np.transpose(points_2d_all, (0, 2, 1, 3))

    _, proj_matrices = _project_point(np.zeros(3), cameras)
    proj_matrices = torch.from_numpy(proj_matrices).float()
    points_2d_all = torch.from_numpy(points_2d_all).float()

    model = UncertaintyWeightedTriangulation(hidden=16)
    X = model(points_2d_all, proj_matrices)
    assert X.shape == (N, J, 3)

    loss = X.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_confidences_weighting():
    """Scalar confidences should downweight low-confidence views."""
    X_true = np.array([1.0, 0.5, 3.0])
    cameras = _make_cameras(4)
    points_2d, proj_matrices = _project_point(X_true, cameras)

    points_2d = points_2d.copy()
    points_2d[0] += np.array([50.0, -30.0])

    points_2d = torch.from_numpy(points_2d).float()
    proj_matrices = torch.from_numpy(proj_matrices).float()

    # Low confidence on the corrupted view.
    confidences = torch.ones(4)
    confidences[0] = 1e-6

    X = triangulate_uncertainty_weighted(
        points_2d, proj_matrices, confidences=confidences
    )
    err = np.linalg.norm(X.detach().numpy() - X_true)
    assert err < 0.1


if __name__ == "__main__":
    test_triangulate_uncertainty_weighted_known_point()
    test_triangulate_uncertainty_weighted_downweights_noisy_view()
    test_triangulate_uncertainty_weighted_is_differentiable()
    test_triangulate_uncertainty_weighted_batched()
    test_uncertainty_weighted_triangulation_module_forward_backward()
    test_confidences_weighting()
    print("uncertainty-weighted triangulation tests passed")
