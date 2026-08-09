"""Unit tests for v52 Uncertainty-Weighted Triangulation.

Tests identity-at-init, masked variable views, and gradient flow through the
precision MLP and weighted DLT.
"""

import pytest
import torch

from motionflow_mv.fusion.uncertainty_weighted_triangulation_v52 import (
    UncertaintyWeightedTriangulationV52,
)
from motionflow_mv.utils.geometry import weighted_dlt_triangulate


def _make_cameras(batch: int = 2, n_views: int = 4, device: str = "cpu"):
    """Return deterministic intrinsics/rotations/translations for testing."""
    K = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(batch, 1, n_views, -1, -1).clone()
    K[:, :, :, 0, 0] = 800.0
    K[:, :, :, 1, 1] = 800.0
    K[:, :, :, 0, 2] = 320.0
    K[:, :, :, 1, 2] = 240.0

    thetas = torch.linspace(0, 2 * torch.pi, n_views + 1, device=device)[:-1]
    R = torch.zeros(batch, 1, n_views, 3, 3, device=device)
    t = torch.zeros(batch, 1, n_views, 3, device=device)
    for i, theta in enumerate(thetas):
        c = torch.tensor([3 * torch.cos(theta), 3 * torch.sin(theta), 1.0], device=device)
        forward = -c / c.norm()
        up = torch.tensor([0.0, 0.0, 1.0], device=device)
        right = torch.linalg.cross(forward, up, dim=0)
        right /= right.norm()
        up = torch.linalg.cross(right, forward, dim=0)
        R[:, :, i] = torch.stack([right, up, -forward], dim=0)
        t[:, :, i] = -(R[:, :, i] @ c)
    K = K.expand(batch, 1, n_views, -1, -1).clone()
    return K, R, t


def _project_points(X_true, K, R, t):
    """Project 3-D points with calibrated cameras.

    Args:
        X_true: (B, T, J, 3).
        K: (B, T, V, 3, 3).
        R: (B, T, V, 3, 3).
        t: (B, T, V, 3).

    Returns:
        points_2d: (B, T, V, J, 2).
    """
    B, T, V, J = X_true.shape[0], X_true.shape[1], K.shape[2], X_true.shape[2]
    P = torch.matmul(K, torch.cat([R, t[..., None]], dim=-1))  # (B, T, V, 3, 4)
    X_h = torch.cat([X_true, torch.ones(B, T, J, 1, device=X_true.device)], dim=-1)  # (B, T, J, 4)
    X_h = X_h.unsqueeze(2)  # (B, T, 1, J, 4)
    x_h = (P.unsqueeze(3) @ X_h.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    points_2d = x_h[..., :2] / (x_h[..., 2:3] + 1e-6)
    return points_2d


def test_weighted_dlt_identity_with_uniform_weights():
    """Uniform weights should give the same result as unweighted DLT."""
    B, T, V, J = 2, 3, 4, 17
    device = "cpu"
    K, R, t = _make_cameras(batch=B, n_views=V, device=device)
    K = K.expand(B, T, V, -1, -1)
    R = R.expand(B, T, V, -1, -1)
    t = t.expand(B, T, V, -1)

    # Synthetic 3D points projected to 2D.
    X_true = torch.randn(B, T, J, 3, device=device)
    points_2d = _project_points(X_true, K, R, t)

    weights = torch.ones(B, T, V, J, device=device)
    X_rec = weighted_dlt_triangulate(points_2d, K, R, t, weights=weights)
    assert X_rec.shape == (B, T, J, 3)
    assert torch.allclose(X_rec, X_true, atol=1e-3)


def test_weighted_dlt_variable_view_mask():
    """Masked-out views should not affect the triangulation result."""
    B, T, V, J = 2, 3, 4, 17
    device = "cpu"
    K, R, t = _make_cameras(batch=B, n_views=V, device=device)
    K = K.expand(B, T, V, -1, -1)
    R = R.expand(B, T, V, -1, -1)
    t = t.expand(B, T, V, -1)

    X_true = torch.randn(B, T, J, 3, device=device)
    points_2d = _project_points(X_true, K, R, t)

    # Mask out the last view.
    view_mask = torch.ones(B, T, V, device=device, dtype=torch.bool)
    view_mask[:, :, -1] = False

    X_rec = weighted_dlt_triangulate(points_2d, K, R, t, view_mask=view_mask)
    assert X_rec.shape == (B, T, J, 3)
    assert torch.allclose(X_rec, X_true, atol=1e-3)
    assert not torch.isnan(X_rec).any()
    assert not torch.isinf(X_rec).any()


def test_uncertainty_weighted_triangulation_identity_at_init():
    """At init, v52 should leave pred_3d_init unchanged (identity property)."""
    B, T, V, J, d = 2, 3, 4, 17, 64
    device = "cpu"
    K, R, t = _make_cameras(batch=B, n_views=V, device=device)
    K = K.expand(B, T, V, -1, -1)
    R = R.expand(B, T, V, -1, -1)
    t = t.expand(B, T, V, -1)

    features = torch.randn(B, T, V, J, d, device=device)
    points_2d = torch.randn(B, T, V, J, 2, device=device)
    pred_3d_init = torch.randn(B, T, J, 3, device=device)
    view_mask = torch.ones(B, T, V, device=device, dtype=torch.bool)

    module = UncertaintyWeightedTriangulationV52(
        d=d,
        n_views=V,
        hidden=64,
        n_layers=2,
        weight_type="per_view_joint",
        identity_init=True,
    )
    pred_3d, uwt_loss, weights, log_precision = module(
        features,
        points_2d,
        K,
        R,
        t,
        pred_3d_init,
        view_mask=view_mask,
    )

    assert pred_3d.shape == (B, T, J, 3)
    assert torch.allclose(pred_3d, pred_3d_init, atol=1e-4)
    assert uwt_loss.numel() == 1
    assert torch.isfinite(uwt_loss)


def test_uncertainty_weighted_triangulation_gradient_flow():
    """Gradients should flow through the precision MLP and weighted DLT."""
    B, T, V, J, d = 2, 3, 4, 17, 64
    device = "cpu"
    K, R, t = _make_cameras(batch=B, n_views=V, device=device)
    K = K.expand(B, T, V, -1, -1)
    R = R.expand(B, T, V, -1, -1)
    t = t.expand(B, T, V, -1)

    features = torch.randn(B, T, V, J, d, device=device, requires_grad=True)
    points_2d = torch.randn(B, T, V, J, 2, device=device)
    pred_3d_init = torch.randn(B, T, J, 3, device=device)
    view_mask = torch.ones(B, T, V, device=device, dtype=torch.bool)

    module = UncertaintyWeightedTriangulationV52(
        d=d,
        n_views=V,
        hidden=64,
        n_layers=2,
        weight_type="per_view_joint",
        identity_init=False,
    )
    pred_3d, uwt_loss, weights, log_precision = module(
        features,
        points_2d,
        K,
        R,
        t,
        pred_3d_init,
        view_mask=view_mask,
    )

    loss = pred_3d.mean() + uwt_loss
    loss.backward()

    assert features.grad is not None
    assert not torch.isnan(features.grad).any()
    # Precision MLP parameters should have non-zero gradients.
    has_param_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for p in module.precision_mlp.parameters()
    )
    assert has_param_grad
