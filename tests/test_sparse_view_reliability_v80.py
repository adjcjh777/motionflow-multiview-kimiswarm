"""Tests for the v80 View-Reliability Before Triangulation module."""

import pytest
import torch
import torch.nn as nn

from motionflow_mv.fusion.sparse_view_reliability_v80 import SparseViewReliabilityV80


def _random_inputs(B=2, T=4, V=4, J=17, d=64, device="cpu"):
    """Generate random but well-formed inputs for SparseViewReliabilityV80."""
    features = torch.randn(B, T, V, J, d, device=device)
    points_2d = torch.randn(B, T, V, J, 2, device=device) * 100.0
    pred_3d = torch.randn(B, T, J, 3, device=device)
    K = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    R = torch.eye(3, device=device).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    t = torch.zeros(B, T, V, 3, device=device)
    return features, points_2d, pred_3d, K, R, t


def test_per_view_output_shape():
    features, points_2d, pred_3d, K, R, t = _random_inputs()
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view")
    out = module(features, points_2d, pred_3d, K, R, t)
    assert out.shape == (2, 4, 4)


def test_per_view_joint_output_shape():
    features, points_2d, pred_3d, K, R, t = _random_inputs()
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view_joint")
    out = module(features, points_2d, pred_3d, K, R, t)
    assert out.shape == (2, 4, 4, 17)


def test_output_values_in_range():
    features, points_2d, pred_3d, K, R, t = _random_inputs()
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view")
    out = module(features, points_2d, pred_3d, K, R, t)
    assert (out > 0.0).all() and (out < 1.0).all()


def test_masked_views_near_zero():
    features, points_2d, pred_3d, K, R, t = _random_inputs()
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view")

    view_mask = torch.ones(2, 4, 4, dtype=torch.float32)
    view_mask[:, :, 0] = 0.0  # mask out first view

    out = module(features, points_2d, pred_3d, K, R, t, view_mask=view_mask)
    assert (out[:, :, 0] < 1e-4).all()
    assert (out[:, :, 1:] > 0.01).all()


def test_per_view_joint_masked_views_near_zero():
    features, points_2d, pred_3d, K, R, t = _random_inputs()
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view_joint")

    view_mask = torch.ones(2, 4, 4, dtype=torch.float32)
    view_mask[:, :, 0] = 0.0

    out = module(features, points_2d, pred_3d, K, R, t, view_mask=view_mask)
    assert (out[:, :, 0, :] < 1e-4).all()
    assert (out[:, :, 1:, :] > 0.01).all()


def test_identity_initialization_mean_near_half():
    features, points_2d, pred_3d, K, R, t = _random_inputs(B=8, T=1, V=4, J=17)
    module = SparseViewReliabilityV80(d=64, n_views=4, n_joints=17, hidden=64, weight_type="per_view")
    out = module(features, points_2d, pred_3d, K, R, t)
    mean = out.mean().item()
    assert 0.45 <= mean <= 0.55
