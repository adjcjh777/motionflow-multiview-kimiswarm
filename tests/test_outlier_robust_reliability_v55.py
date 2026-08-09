"""Unit tests for v55 Outlier-Robust Reliability (OR2)."""

import pytest
import torch

from motionflow_mv.fusion.outlier_robust_reliability_v55 import (
    OutlierRobustReliabilityV55,
)


@pytest.fixture
def toy_inputs():
    B, T, V, J, d = 2, 3, 4, 17, 64
    features = torch.randn(B, T, V, J, d)
    points_2d = torch.rand(B, T, V, J, 2) * 512
    K = torch.eye(3, dtype=torch.float).reshape(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    K[:, :, :, 0, 0] = 500.0
    K[:, :, :, 1, 1] = 500.0
    K[:, :, :, 0, 2] = 256.0
    K[:, :, :, 1, 2] = 256.0
    R = torch.eye(3, dtype=torch.float).reshape(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    t = torch.zeros(B, T, V, 3)
    pred_3d_init = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    return features, points_2d, K, R, t, pred_3d_init, view_mask


def test_identity_at_init(toy_inputs):
    features, points_2d, K, R, t, pred_3d_init, view_mask = toy_inputs
    module = OutlierRobustReliabilityV55(d=64, n_views=4, identity_init=True)
    weights, loss = module(
        features,
        points_2d,
        K,
        R,
        t,
        pred_3d_init,
        view_mask=view_mask,
    )
    assert weights.shape == features.shape[:4]
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-3)
    assert loss.item() >= 0.0


def test_weight_clamping(toy_inputs):
    features, points_2d, K, R, t, pred_3d_init, view_mask = toy_inputs
    module = OutlierRobustReliabilityV55(d=64, n_views=4, min_weight=0.05)
    weights, _ = module(
        features,
        points_2d,
        K,
        R,
        t,
        pred_3d_init,
        view_mask=view_mask,
    )
    assert weights.min() >= 0.05 - 1e-6
    assert weights.max() <= 1.0 + 1e-6


def test_gradient_flow(toy_inputs):
    features, points_2d, K, R, t, pred_3d_init, view_mask = toy_inputs
    module = OutlierRobustReliabilityV55(d=64, n_views=4, identity_init=False)
    weights, loss = module(
        features,
        points_2d,
        K,
        R,
        t,
        pred_3d_init.requires_grad_(True),
        view_mask=view_mask,
    )
    loss.backward()
    assert pred_3d_init.grad is not None
    assert pred_3d_init.grad.abs().sum() > 0.0


def test_synthetic_outlier_downweight(toy_inputs):
    features, points_2d, K, R, t, pred_3d_init, view_mask = toy_inputs
    # Corrupt one view's 2D observations.
    points_2d_corrupted = points_2d.clone()
    points_2d_corrupted[:, :, 0, :, :] += 200.0
    module = OutlierRobustReliabilityV55(d=64, n_views=4, identity_init=False)
    weights, _ = module(
        features,
        points_2d_corrupted,
        K,
        R,
        t,
        pred_3d_init,
        view_mask=view_mask,
    )
    # The corrupted view should, on average, receive lower weight.
    mean_clean = weights[:, :, 1:, :].mean()
    mean_corrupt = weights[:, :, 0, :].mean()
    assert mean_corrupt < mean_clean


def test_view_mask_zeroes_weights():
    B, T, V, J, d = 1, 2, 4, 17, 64
    features = torch.randn(B, T, V, J, d)
    points_2d = torch.rand(B, T, V, J, 2)
    K = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(B, T, V, -1, -1)
    R = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(B, T, V, -1, -1)
    t = torch.zeros(B, T, V, 3)
    pred_3d_init = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    view_mask[:, :, 0] = False
    module = OutlierRobustReliabilityV55(d=64, n_views=4)
    weights, _ = module(features, points_2d, K, R, t, pred_3d_init, view_mask=view_mask)
    assert weights[:, :, 0, :].sum().item() == 0.0
