"""Unit tests for the Iskakov et al. (ICCV 2019) learnable-triangulation baseline.

Covers:
* forward output shape (N, J, 3) for (N, V, J, 2) inputs;
* zero-initialisation equivalence: at init the model's final MLP layer is
  zero, so all predicted weights equal 0.5 and the output must match the
  unweighted batched DLT solve within 1e-4;
* gradient flow through the differentiable weighted-lstsq solve;
* ``build_projection_matrices`` shape and camera-model correctness;
* ``build_features`` shape / finiteness;
* ``return_weights`` path and weight range (0, 1).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.iskakov_learnable_triangulation import (
    FEATURE_NAMES,
    NUM_FEATURES,
    IskakovLearnableTriangulation,
    build_features,
    build_projection_matrices,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq


def _random_cameras(V: int, seed: int = 0):
    """Return (K, R, t) tensors for *V* synthetic cameras."""
    rng = np.random.default_rng(seed)
    K = torch.zeros(V, 3, 3, dtype=torch.float64)
    R = torch.zeros(V, 3, 3, dtype=torch.float64)
    t = torch.zeros(V, 3, dtype=torch.float64)
    for v in range(V):
        Kk = np.eye(3)
        Kk[0, 0] = Kk[1, 1] = float(rng.uniform(700, 900))
        Kk[:2, 2] = rng.uniform(300, 340, size=2)
        Rk, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Rk) < 0:
            Rk[:, 0] *= -1
        tk = rng.standard_normal(3) * 2.0
        K[v] = torch.from_numpy(Kk)
        R[v] = torch.from_numpy(Rk)
        t[v] = torch.from_numpy(tk)
    return K, R, t


def _project_points(X: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    """Project (N, J, 3) world points through P = K[R|t] -> (N, V, J, 2)."""
    N, J, _ = X.shape
    V = K.shape[0]
    P = build_projection_matrices(K, R, t)  # (V, 3, 4)
    X_h = torch.cat([X, torch.ones(N, J, 1, dtype=X.dtype)], dim=-1)  # (N, J, 4)
    uv = torch.einsum("vab,njb->nvja", P, X_h)  # (N, V, J, 3)
    return uv[..., :2] / uv[..., 2:3].clamp(min=1e-6)


@pytest.fixture
def setup():
    torch.manual_seed(0)
    N, V, J = 2, 3, 17
    K, R, t = _random_cameras(V, seed=1)
    X = torch.randn(N, J, 3, dtype=torch.float64) * 0.3
    X[..., 2] += 1.0
    points_2d = _project_points(X, K, R, t)
    confidences = torch.full((N, V, J), 0.9, dtype=torch.float64)
    return points_2d, confidences, K, R, t, X


def test_forward_shape(setup):
    points_2d, conf, K, R, t, _ = setup
    model = IskakovLearnableTriangulation(hidden_dim=32, cross_view=True).double()
    X = model(points_2d, conf, K, R, t)
    assert X.shape == (points_2d.shape[0], points_2d.shape[2], 3)
    assert torch.isfinite(X).all()


def test_zero_init_matches_unweighted_dlt(setup):
    """Final layer is zero-initialised -> weights are all 0.5 -> the model must
    reproduce the unweighted batched DLT solve."""
    points_2d, conf, K, R, t, _ = setup
    model = IskakovLearnableTriangulation(hidden_dim=32, cross_view=True).double()
    X_model = model(points_2d, conf, K, R, t)
    P = build_projection_matrices(K, R, t)
    X_dlt = triangulate_dlt_batched_lstsq(points_2d, P)
    torch.testing.assert_close(X_model, X_dlt, atol=1e-4, rtol=1e-6)


def test_gradient_flows_through_lstsq(setup):
    points_2d, conf, K, R, t, X_gt = setup
    model = IskakovLearnableTriangulation(hidden_dim=32, cross_view=True).double()
    # Perturb the final layer away from zero so weights actually depend on it.
    with torch.no_grad():
        model.mlp[-1].weight.add_(0.1)
    X = model(points_2d, conf, K, R, t)
    loss = torch.mean(torch.norm(X - X_gt, dim=-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients propagated through the weighted DLT solve"
    assert all(torch.isfinite(g).all() for g in grads)
    assert model.mlp[-1].weight.grad.abs().sum() > 0


def test_projection_matrices_shape_and_model(setup):
    points_2d, conf, K, R, t, X = setup
    P = build_projection_matrices(K, R, t)
    assert P.shape == (K.shape[0], 3, 4)
    # Reprojecting the 3D points through P must recover the 2D detections.
    uv = _project_points(X, K, R, t)
    torch.testing.assert_close(uv, points_2d, atol=1e-6, rtol=1e-6)


def test_build_features_shape_and_finiteness(setup):
    points_2d, conf, K, R, t, _ = setup
    feats = build_features(points_2d, conf, K, R, t)
    assert feats.shape == points_2d.shape[:3] + (NUM_FEATURES,)
    assert torch.isfinite(feats).all()
    assert NUM_FEATURES == len(FEATURE_NAMES)


def test_return_weights_range(setup):
    points_2d, conf, K, R, t, _ = setup
    model = IskakovLearnableTriangulation(hidden_dim=32, cross_view=True).double()
    X, w = model(points_2d, conf, K, R, t, return_weights=True)
    assert w.shape == points_2d.shape[:3]
    assert (w > 0).all() and (w < 1).all()
    # At init all weights are exactly 0.5 (zero final layer).
    torch.testing.assert_close(w, torch.full_like(w, 0.5), atol=1e-6, rtol=0)


def test_per_view_variant_forward(setup):
    points_2d, conf, K, R, t, _ = setup
    model = IskakovLearnableTriangulation(hidden_dim=32, cross_view=False).double()
    X = model(points_2d, conf, K, R, t)
    assert X.shape == (points_2d.shape[0], points_2d.shape[2], 3)
    assert torch.isfinite(X).all()
