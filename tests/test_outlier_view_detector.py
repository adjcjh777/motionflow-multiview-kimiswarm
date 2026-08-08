"""Unit tests for motionflow_mv/fusion/outlier_view_detector.py.

Covers the public contract of ``OutlierViewDetector``:
- forward shape ``(B, T, V, J, 2) -> (B, T, V, J)``
- identity-at-init (all-one weights on clean views)
- down-weighting of corrupted views
- view-masking behaviour
- gradient flow through inputs
- helper ``compute_reprojection_residual`` shape and monotonicity
"""

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.outlier_view_detector import (
    OutlierViewDetector,
    compute_reprojection_residual,
)


def _make_cameras(n_views: int = 4):
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


def _project_points(joints_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project world points into all views; returns (T, V, J, 2)."""
    # joints_3d: (T, J, 3); R: (V, 3, 3); t: (V, 3); K: (V, 3, 3)
    X_cam = torch.einsum("vab,tjb->tvja", R, joints_3d) + t[None, :, None, :]
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[None, :, None, :, :], (X_cam / z)[..., None]).squeeze(-1)
    points_2d = uv[..., :2] / uv[..., 2:3]
    return points_2d


def _make_batch(B: int = 2, T: int = 3, V: int = 4, J: int = 17):
    """Return synthetic (X, points_2d, K, R, t, view_mask)."""
    K, R, t = _make_cameras(V)
    torch.manual_seed(42)
    joints_3d = torch.randn(T, J, 3) * 0.3
    points_2d = _project_points(joints_3d, K, R, t)  # (T, V, J, 2)
    points_2d = points_2d.unsqueeze(0).expand(B, -1, -1, -1, -1)

    K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    t = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    X = joints_3d.unsqueeze(0).expand(B, -1, -1, -1)
    view_mask = torch.ones(B, T, V).bool()
    return X, points_2d, K, R, t, view_mask


# ---------------------------------------------------------------------------
# Shape / helper tests
# ---------------------------------------------------------------------------

def test_compute_reprojection_residual_shape():
    X, points_2d, K, R, t, _ = _make_batch()
    residual = compute_reprojection_residual(X, points_2d, K, R, t)
    B, T, V, J = points_2d.shape[:4]
    assert residual.shape == (B, T, V, J)


def test_forward_shape():
    detector = OutlierViewDetector()
    X, points_2d, K, R, t, view_mask = _make_batch()
    weights, residual = detector(X, points_2d, K, R, t, view_mask=view_mask)
    B, T, V, J = points_2d.shape[:4]
    assert weights.shape == (B, T, V, J)
    assert residual.shape == (B, T, V, J)


# ---------------------------------------------------------------------------
# Identity at init
# ---------------------------------------------------------------------------

def test_identity_at_init_clean_views():
    detector = OutlierViewDetector()
    X, points_2d, K, R, t, view_mask = _make_batch()
    weights, _ = detector(X, points_2d, K, R, t, view_mask=view_mask)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5)


# ---------------------------------------------------------------------------
# Outlier down-weighting
# ---------------------------------------------------------------------------

def test_downweights_corrupted_view():
    detector = OutlierViewDetector(z_thresh=2.0, soft_beta=2.0)
    X, points_2d, K, R, t, view_mask = _make_batch()

    # Corrupt view 0 for all joints.
    points_2d_corrupt = points_2d.clone()
    points_2d_corrupt[:, :, 0, :, :] += 100.0

    weights, _ = detector(X, points_2d_corrupt, K, R, t, view_mask=view_mask)
    # Average weight for the corrupted view should be lower than clean views.
    assert weights[:, :, 0, :].mean() < weights[:, :, 1, :].mean()


def test_no_false_positives_on_low_noise():
    detector = OutlierViewDetector(z_thresh=3.0, soft_beta=1.0)
    X, points_2d, K, R, t, view_mask = _make_batch()
    # Add small Gaussian noise to one view.
    torch.manual_seed(123)
    points_2d_noisy = points_2d.clone()
    points_2d_noisy[:, :, 0, :, :] += torch.randn_like(points_2d_noisy[:, :, 0, :, :]) * 0.5
    weights, _ = detector(X, points_2d_noisy, K, R, t, view_mask=view_mask)
    # No view should be strongly down-weighted with mild noise.
    assert weights.min() > 0.5


# ---------------------------------------------------------------------------
# View masking
# ---------------------------------------------------------------------------

def test_view_mask_zeros_masked_views():
    detector = OutlierViewDetector()
    X, points_2d, K, R, t, _ = _make_batch()
    view_mask = torch.ones(2, 3, 4).bool()
    view_mask[:, :, -1] = False
    weights, _ = detector(X, points_2d, K, R, t, view_mask=view_mask)
    assert weights[:, :, -1, :].sum() == 0.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    detector = OutlierViewDetector()
    X, points_2d, K, R, t, view_mask = _make_batch()
    X.requires_grad_(True)
    points_2d.requires_grad_(True)
    K.requires_grad_(True)
    R.requires_grad_(True)
    t.requires_grad_(True)

    weights, residual = detector(X, points_2d, K, R, t, view_mask=view_mask)
    loss = weights.mean() + residual.mean()
    loss.backward()

    assert points_2d.grad is not None
    assert K.grad is not None
    assert R.grad is not None
    assert t.grad is not None
    assert X.grad is not None


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,T,V,J", [(1, 1, 4, 17), (2, 3, 6, 28)])
def test_varied_geometry(B, T, V, J):
    detector = OutlierViewDetector()
    X, points_2d, K, R, t, view_mask = _make_batch(B, T, V, J)
    weights, residual = detector(X, points_2d, K, R, t, view_mask=view_mask)
    assert weights.shape == (B, T, V, J)
    assert (weights >= 0.0).all() and (weights <= 1.0).all()


if __name__ == "__main__":
    test_compute_reprojection_residual_shape()
    test_forward_shape()
    test_identity_at_init_clean_views()
    test_downweights_corrupted_view()
    test_no_false_positives_on_low_noise()
    test_view_mask_zeros_masked_views()
    test_gradient_flow()
    test_varied_geometry(1, 1, 4, 17)
    test_varied_geometry(2, 3, 6, 28)
    print("All OutlierViewDetector tests passed")
