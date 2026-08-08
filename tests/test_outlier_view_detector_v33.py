import math

import pytest
import torch

from motionflow_mv.fusion.outlier_view_detector_v33 import OutlierViewDetectorV33


def _make_cameras(n_views: int = 4, b: int = 2, t: int = 3, j: int = 17):
    """Return deterministic intrinsics/extrinsics."""
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, n_views, 3, 3).clone()
    K[:, :, :, 0, 0] = 800.0
    K[:, :, :, 1, 1] = 800.0
    K[:, :, :, 0, 2] = 320.0
    K[:, :, :, 1, 2] = 240.0

    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, n_views, 3, 3).clone()
    t_vec = torch.zeros(b, t, n_views, 3)
    for v in range(n_views):
        theta = 2 * math.pi * v / n_views
        t_vec[:, :, v, 0] = 3.0 * math.cos(theta)
        t_vec[:, :, v, 1] = 3.0 * math.sin(theta)
        t_vec[:, :, v, 2] = 0.0

    return K, R, t_vec


def test_outlier_detector_identity_at_init():
    b, t, v, j = 2, 3, 4, 17
    detector = OutlierViewDetectorV33(num_joints=j, num_parts=5, num_domains=3, use_feature_gate=True)
    K, R, t_vec = _make_cameras(v, b, t, j)
    pred_3d = torch.randn(b, t, j, 3)
    points_2d = torch.randn(b, t, v, j, 2)
    features = torch.randn(b, t, v, j, 64)

    weights, aux_loss = detector(
        pred_3d, points_2d, K, R, t_vec, features=features, domain_ids=None, view_mask=None, outlier_label=None
    )
    assert weights.shape == (b, t, v, j)
    assert weights.min() > 0.99  # identity at init
    assert aux_loss.item() == 0.0


def test_outlier_detector_detects_outlier():
    b, t, v, j = 2, 1, 4, 17
    detector = OutlierViewDetectorV33(num_joints=j, use_feature_gate=False)
    # Open the gate to test the actual outlier-down-weight behaviour.
    detector.residual_scale.data = torch.tensor(6.0)
    # Cameras at the origin, facing the +Z direction: true projection of (0,0,5) is (320, 240).
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    K[:, :, :, 0, 0] = 800.0
    K[:, :, :, 1, 1] = 800.0
    K[:, :, :, 0, 2] = 320.0
    K[:, :, :, 1, 2] = 240.0
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, v, 3, 3).clone()
    t_vec = torch.zeros(b, t, v, 3)
    pred_3d = torch.zeros(b, t, j, 3)
    pred_3d[..., 2] = 5.0
    points_2d = torch.full((b, t, v, j, 2), fill_value=0.0)
    points_2d[..., 0] = 320.0
    points_2d[..., 1] = 240.0
    # Make view 0 a strong outlier.
    points_2d[:, :, 0, :, 0] += 100.0
    weights, _ = detector(pred_3d, points_2d, K, R, t_vec, features=None, outlier_label=None)
    assert weights[:, :, 0, :].mean() < weights[:, :, 1:, :].mean()
