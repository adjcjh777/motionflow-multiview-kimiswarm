"""Unit tests for v45 Adaptive Geometry Fusion.

These tests do not start any GPU training; they only verify that the module
produces sensible shapes and positive weights, and that the weights can be used
in a weighted DLT triangulation.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.adaptive_geometry_fusion_v45 import AdaptiveGeometryFusionV45
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    MultiViewGeometryFusionV25,
    triangulate_initial,
)


def _make_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras."""
    import numpy as np

    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


def _project_points(joints_3d, K, R, t):
    """Project world points into each view.

    Args:
        joints_3d: (B, T, J, 3).
        K, R, t: (V, 3, 3), (V, 3, 3), (V, 3).

    Returns:
        points_2d: (B, T, V, J, 2).
    """
    V = K.shape[0]
    # joints_3d: (B, T, J, 3) -> expand to (B, T, V, J, 3).
    X = joints_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)
    # X in world: (B, T, V, J, 3); R is (V, 3, 3).
    X = X.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
    X_cam = torch.matmul(R, X) + t[..., None]  # (B, T, V, 3, J)
    X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[..., None, :, :], (X_cam / z)[..., None]).squeeze(-1)
    return uv[..., :2] / uv[..., 2:3]


@pytest.fixture(params=["per_view", "per_joint", "per_view_joint"])
def weight_type(request):
    return request.param


@pytest.fixture
def dummy_scene():
    B, T, V, J = 2, 3, 4, 17
    K, R, t = _make_cameras(V)
    joints_3d = torch.randn(B, T, J, 3) * 0.3
    points_2d = _project_points(joints_3d, K, R, t)
    return points_2d, joints_3d, K, R, t


def test_module_forward_shape_and_positivity(weight_type, dummy_scene):
    points_2d, pred_3d, K, R, t = dummy_scene
    B, T, V, J = points_2d.shape[:4]
    module = AdaptiveGeometryFusionV45(
        n_views=V,
        weight_type=weight_type,
        hidden=16,
        n_layers=2,
    )
    weights = module(points_2d, pred_3d, K, R, t)
    assert weights.shape == (B, T, V, J)
    assert weights.min().item() > 0.0
    assert weights.max().item() < 1e4


def test_module_view_mask(weight_type, dummy_scene):
    points_2d, pred_3d, K, R, t = dummy_scene
    B, T, V, J = points_2d.shape[:4]
    module = AdaptiveGeometryFusionV45(
        n_views=V,
        weight_type=weight_type,
        hidden=16,
        n_layers=1,
    )
    view_mask = torch.ones(B, T, V)
    view_mask[:, :, -1] = 0.0
    weights = module(points_2d, pred_3d, K, R, t, view_mask=view_mask)
    assert weights[..., :-1, :].max().item() > 0.0
    # Masked views are clamped to a small positive floor for numerical
    # stability in the weighted DLT step.
    assert weights[..., -1, :].max().item() <= 1e-4


def test_module_initial_weights_near_one(dummy_scene):
    points_2d, pred_3d, K, R, t = dummy_scene
    module = AdaptiveGeometryFusionV45(n_views=4, weight_type="per_view")
    weights = module(points_2d, pred_3d, K, R, t)
    # Zero-initialised final layer -> 2 * sigmoid(0) = 1.0.
    assert torch.allclose(weights, torch.ones_like(weights), atol=0.1)


def test_v25_integration(dummy_scene):
    """MultiViewGeometryFusionV25 with v45 enabled should run and return a loss."""
    points_2d, pred_3d_init, K, R, t = dummy_scene
    B, T, V, J = points_2d.shape[:4]
    confidence = torch.ones(B, T, V, J)
    model = MultiViewGeometryFusionV25(
        d=32,
        n_views=V,
        n_heads=2,
        n_geometry_layers=1,
        use_geometry_attention=False,
        use_learned_depth_triangulation=False,
        use_outlier_view_detector=False,
        use_v45_adaptive_geometry_fusion=True,
        v45_adaptive_weight_type="per_view",
    )
    pred_3d_ref, geom_loss = model(
        points_2d=points_2d,
        K=K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1),
        R=R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1),
        t=t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
        pred_3d_init=pred_3d_init,
        confidence=confidence,
    )
    assert pred_3d_ref.shape == (B, T, J, 3)
    assert geom_loss.numel() == 1


def test_weights_sum_over_views(dummy_scene):
    """Per-view weights should sum to a sensible positive value after masking."""
    points_2d, pred_3d, K, R, t = dummy_scene
    B, T, V, J = points_2d.shape[:4]
    module = AdaptiveGeometryFusionV45(n_views=V, weight_type="per_view")
    weights = module(points_2d, pred_3d, K, R, t)
    sum_per_frame = weights.sum(dim=-2)  # over views
    assert sum_per_frame.min().item() > 0.0
    assert sum_per_frame.shape == (B, T, J)


def test_triangulation_with_v45_weights(dummy_scene):
    """v45 weights can be fed to the batched DLT triangulation."""
    points_2d, pred_3d, K, R, t = dummy_scene
    B, T, V, J = points_2d.shape[:4]
    module = AdaptiveGeometryFusionV45(n_views=V, weight_type="per_joint")
    weights = module(points_2d, pred_3d, K, R, t)
    K_bt = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    R_bt = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    t_bt = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    X = triangulate_initial(points_2d, K_bt, R_bt, t_bt, weights=weights)
    assert X.shape == (B, T, J, 3)
