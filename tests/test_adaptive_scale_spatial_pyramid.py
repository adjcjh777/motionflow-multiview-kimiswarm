"""Sanity tests for the adaptive scale-gated cross-view spatial pyramid."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.cross_view_spatial_pyramid import (
    AdaptiveScaleCrossViewSpatialPyramid,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_adaptive_scale_pyramid_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveScalePyramid,
)
from motionflow_mv.fusion.ray_attention_temporal_model import _make_cameras


def test_adaptive_scale_pyramid_forward_shape_and_grad():
    B, T, V, J = 2, 5, 4, 17
    x = torch.rand(B, T, V, J, 3)
    cameras = _make_cameras(V)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveScalePyramid(
        j=J,
        d=64,
        n_views=V,
        n_heads=4,
        n_joint_layers=1,
        n_st_layers=2,
        residual_hidden=128,
        return_pp_delta=True,
    )

    pred, weights, _ = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_adaptive_scale_pyramid_module_shapes():
    N, V, J, d = 2, 4, 17, 32
    x = torch.rand(N, V, J, d)
    pyramid = AdaptiveScaleCrossViewSpatialPyramid(d=d, n_views=V, scales=(1, 2, 4))
    out = pyramid(x)
    assert out.shape == (N, V, J, d)


def test_adaptive_scale_pyramid_single_scale():
    N, V, J, d = 2, 4, 17, 32
    x = torch.rand(N, V, J, d)
    pyramid = AdaptiveScaleCrossViewSpatialPyramid(d=d, n_views=V, scales=(1,))
    out = pyramid(x)
    assert out.shape == (N, V, J, d)


if __name__ == "__main__":
    test_adaptive_scale_pyramid_forward_shape_and_grad()
    test_adaptive_scale_pyramid_module_shapes()
    test_adaptive_scale_pyramid_single_scale()
    print("adaptive scale-gated spatial pyramid tests passed")
