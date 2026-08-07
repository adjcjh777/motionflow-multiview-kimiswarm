"""Extended unit tests for the Bayesian tri v2 batched DLT.

These tests exercise ``motionflow_mv.fusion.triangulation.triangulate_dlt_batched_lstsq``
as used by ``RayAttentionFusionModelBayesianTriV2``.  They complement the
original ``tests/test_bayesian_tri_v2_batched_dlt.py`` with additional edge
cases, dtype/device handling, and a lightweight end-to-end model smoke test.
"""

import numpy as np
import pytest
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)
from motionflow_mv.fusion.triangulation import (
    triangulate_dlt,
    triangulate_dlt_batched_lstsq,
    triangulate_dlt_torch,
)


def _make_cameras(n_views: int = 4, radius: float = 3.0, seed: int = 42):
    """Create a deterministic circular multi-view rig."""
    rng = np.random.default_rng(seed)
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([radius * np.cos(theta), radius * np.sin(theta), 1.0])
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


def _numpy_lstsq_reference(points_2d, proj_matrices, weights):
    rows = []
    for (u, v), P, weight in zip(points_2d, proj_matrices, weights):
        scale = np.sqrt(weight + 1e-6)
        rows.extend(
            [
                scale * (u * P[2] - P[0]),
                scale * (v * P[2] - P[1]),
            ]
        )
    A = np.stack(rows)
    return np.linalg.lstsq(A[:, :3], -A[:, 3], rcond=None)[0]


def test_batched_dlt_v2_minimum_two_views():
    """DLT should recover a point from only two views."""
    cameras = _make_cameras(n_views=2)
    rng = np.random.default_rng(7)
    points_3d = rng.uniform(-1.0, 1.0, (1, 1, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices)
    np.testing.assert_allclose(pred[0, 0].detach().numpy(), points_3d[0, 0], atol=1e-7)


def test_batched_dlt_v2_proj_matrices_unbatched():
    """Passing a single (V, 3, 4) projection matrix should be expanded."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(13)
    points_3d = rng.uniform(-1.0, 1.0, (3, 5, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)

    pred_batched = triangulate_dlt_batched_lstsq(
        points_2d, proj_matrices.unsqueeze(0).expand(len(points_2d), -1, -1, -1)
    )
    pred_unbatched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices)

    torch.testing.assert_close(pred_batched, pred_unbatched, atol=1e-8, rtol=1e-8)


def test_batched_dlt_v2_batch_joint_independence():
    """Different (batch, joint) points should be reconstructed independently."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(2027)
    points_3d = rng.uniform(-1.0, 1.0, (4, 6, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)
    weights = torch.rand(points_2d.shape[:3], dtype=torch.float64)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    pred_np = pred.detach().numpy()

    np.testing.assert_allclose(pred_np, points_3d, atol=1e-8)


def test_batched_dlt_v2_float32():
    """Smoke test for float32 inputs and reasonable single-precision accuracy."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(99)
    points_3d = rng.uniform(-1.0, 1.0, (2, 5, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float32)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float32)
    weights = torch.rand(points_2d.shape[:3], dtype=torch.float32)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    np.testing.assert_allclose(pred.detach().numpy(), points_3d, atol=1e-4)


def test_batched_dlt_v2_unity_weights_match_no_weights():
    """All-ones weights should give the same result as omitting weights."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(55)
    points_3d = rng.uniform(-1.0, 1.0, (2, 5, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)
    weights = torch.ones(points_2d.shape[:3], dtype=torch.float64)

    pred_no_w = triangulate_dlt_batched_lstsq(points_2d, proj_matrices)
    pred_w = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)

    torch.testing.assert_close(pred_no_w, pred_w, atol=1e-10, rtol=1e-10)


def test_batched_dlt_v2_zero_weights_remain_finite():
    """Zero weights should not produce NaNs/Inf and gradients should remain finite."""
    N, V, J = 2, 4, 6
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64, requires_grad=True)
    proj_matrices = torch.randn(V, 3, 4, dtype=torch.float64)
    weights = torch.zeros(N, V, J, dtype=torch.float64, requires_grad=True)

    pred = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    assert torch.isfinite(pred).all()
    loss = pred.sum()
    loss.backward()
    assert points_2d.grad is not None and torch.isfinite(points_2d.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_batched_dlt_v2_matches_per_joint_torch_broadcasted():
    """Batched result should match per-joint ``triangulate_dlt_torch`` with broadcasting."""
    N, V, J = 2, 4, 7
    rng = np.random.default_rng(11)
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64)
    proj_matrices = torch.randn(V, 3, 4, dtype=torch.float64)
    weights = torch.rand(N, V, J, dtype=torch.float64)

    pred_batched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)

    pred_single = torch.zeros(N, J, 3, dtype=torch.float64)
    for j in range(J):
        pred_single[:, j, :] = triangulate_dlt_torch(
            points_2d[:, :, j, :], proj_matrices, weights[:, :, j]
        )

    torch.testing.assert_close(pred_batched, pred_single, atol=1e-5, rtol=1e-5)


def test_batched_dlt_v2_device_consistency():
    """CPU and CUDA (if available) should produce identical results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    N, V, J = 2, 4, 5
    points_2d = torch.rand(N, V, J, 2, dtype=torch.float64)
    proj_matrices = torch.randn(V, 3, 4, dtype=torch.float64)
    weights = torch.rand(N, V, J, dtype=torch.float64)

    pred_cpu = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)
    pred_cuda = triangulate_dlt_batched_lstsq(
        points_2d.cuda(), proj_matrices.cuda(), weights.cuda()
    )
    torch.testing.assert_close(pred_cpu, pred_cuda.cpu(), atol=1e-7, rtol=1e-7)


def test_batched_dlt_v2_matches_naive_numpy():
    """Batched result should match the numpy reference for each point individually."""
    cameras = _make_cameras(4)
    rng = np.random.default_rng(31)
    points_3d = rng.uniform(-1.0, 1.0, (3, 4, 3))

    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d = torch.from_numpy(points_2d_np).to(torch.float64)
    proj_matrices = torch.from_numpy(proj_matrices_np).to(torch.float64)
    weights = torch.rand(points_2d.shape[:3], dtype=torch.float64)

    pred_batched = triangulate_dlt_batched_lstsq(points_2d, proj_matrices, weights)

    for n in range(points_2d.shape[0]):
        for j in range(points_2d.shape[2]):
            pred_np = triangulate_dlt(
                points_2d_np[n, :, j, :],
                proj_matrices_np,
                weights[n, :, j].detach().numpy(),
            )
            np.testing.assert_allclose(
                pred_batched[n, j].detach().numpy(), pred_np, atol=1e-7
            )


def test_batched_dlt_v2_routes_weights_per_batch():
    B, V, J = 2, 4, 2
    cameras = _make_cameras(V)
    points_3d = np.random.default_rng(31).uniform(-1.0, 1.0, (B, J, 3))
    points_2d_np, proj_matrices_np = _project_points(points_3d, cameras)
    points_2d_np += np.random.default_rng(731).normal(0.0, 2.0, points_2d_np.shape)
    patterns = np.array(
        [
            [1.0, 1.0, 1e-3, 1e-3],
            [1e-3, 1e-3, 1.0, 1.0],
        ]
    )
    weights_np = np.repeat(patterns[:, :, None], J, axis=2)

    pred = triangulate_dlt_batched_lstsq(
        torch.from_numpy(points_2d_np).to(torch.float64),
        torch.from_numpy(proj_matrices_np).to(torch.float64),
        torch.from_numpy(weights_np).to(torch.float64),
    ).detach().numpy()

    for n in range(B):
        for j in range(J):
            expected = _numpy_lstsq_reference(
                points_2d_np[n, :, j], proj_matrices_np, weights_np[n, :, j]
            )
            np.testing.assert_allclose(pred[n, j], expected, atol=1e-8, rtol=1e-8)

    correct = _numpy_lstsq_reference(
        points_2d_np[1, :, 0], proj_matrices_np, weights_np[1, :, 0]
    )
    wrong_batch = _numpy_lstsq_reference(
        points_2d_np[1, :, 0], proj_matrices_np, weights_np[0, :, 0]
    )
    assert np.linalg.norm(correct - wrong_batch) > 1e-3


def test_bayesian_tri_v2_model_forward_backward_smoke():
    """The full BayesianTriV2 model should run forward/backward using batched DLT."""
    B, T, V, J = 2, 3, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV2(
        j=J,
        d=32,
        n_views=V,
        n_heads=2,
        n_st_layers=1,
        n_joint_layers=1,
        gn_iters=1,
        epipolar_loss_weight=0.0,
        return_pp_delta=True,
    )
    pred, weights, pp_delta, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert epi_loss.shape == ()
    loss = pred.mean() + 0.0 * epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_bayesian_tri_v2_model_squeeze_output():
    """Passing a 4-D input should be squeezed back to 3-D outputs."""
    V, J = 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(2, V, J, 3)
    model = RayAttentionFusionModelBayesianTriV2(
        j=J,
        d=32,
        n_views=V,
        n_heads=2,
        n_st_layers=1,
        n_joint_layers=1,
        gn_iters=1,
        epipolar_loss_weight=0.0,
    )
    pred, weights, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (2, J, 3)
    assert weights.shape == (2, V, J)
    assert epi_loss.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
