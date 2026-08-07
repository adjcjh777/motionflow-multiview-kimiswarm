"""Geometry contract for per-view camera-centric ray-depth scaling."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_camera_centric_model import (
    _apply_per_view_ray_depth_scale,
)


def _project(points_3d, K, R, t):
    camera = torch.einsum("bvij,bkj->bvki", R, points_3d) + t[:, :, None, :]
    image = torch.einsum("bvij,bvkj->bvki", K, camera)
    return image[..., :2] / image[..., 2:3]


def test_ray_depth_scale_identity_and_translation_equivariance():
    dtype = torch.float64
    points_3d = torch.tensor([[[2.0, 3.0, 5.0], [1.0, -2.0, 4.0]]], dtype=dtype)
    camera_centers = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 1.0, 0.0]]], dtype=dtype)
    R = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    t = -camera_centers
    K = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    points_2d = _project(points_3d, K, R, t)
    weights = torch.tensor([[[1.0, 0.25], [0.5, 1.0]]], dtype=dtype)

    identity = _apply_per_view_ray_depth_scale(
        points_3d,
        points_2d,
        torch.ones(1, 2, dtype=dtype),
        weights,
        K,
        R,
        t,
    )
    torch.testing.assert_close(identity, points_3d, rtol=0, atol=0)

    scale = torch.tensor([[0.8, 1.1]], dtype=dtype)
    output = _apply_per_view_ray_depth_scale(
        points_3d, points_2d, scale, weights, K, R, t
    )
    shift = torch.tensor([[[10.0, -4.0, 2.0]]], dtype=dtype)
    shifted_output = _apply_per_view_ray_depth_scale(
        points_3d + shift,
        points_2d,
        scale,
        weights,
        K,
        R,
        t - torch.einsum("bvij,bkj->bvki", R, shift).squeeze(2),
    )
    torch.testing.assert_close(shifted_output, output + shift, rtol=0, atol=1e-12)


def test_ray_depth_scale_preserves_per_view_information():
    dtype = torch.float64
    points_3d = torch.tensor([[[2.0, 3.0, 5.0]]], dtype=dtype)
    camera_centers = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 1.0, 0.0]]], dtype=dtype)
    R = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    t = -camera_centers
    K = torch.eye(3, dtype=dtype).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    points_2d = _project(points_3d, K, R, t)
    weights = torch.ones(1, 2, 1, dtype=dtype)

    first = _apply_per_view_ray_depth_scale(
        points_3d,
        points_2d,
        torch.tensor([[0.8, 1.2]], dtype=dtype),
        weights,
        K,
        R,
        t,
    )
    swapped = _apply_per_view_ray_depth_scale(
        points_3d,
        points_2d,
        torch.tensor([[1.2, 0.8]], dtype=dtype),
        weights,
        K,
        R,
        t,
    )
    assert not torch.allclose(first, swapped, rtol=0, atol=1e-12)
