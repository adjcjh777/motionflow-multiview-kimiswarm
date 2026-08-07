"""Scale-weight diagnostics for the adaptive temporal pyramid."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_adaptive_window_pyramid_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid,
)


def test_adaptive_pyramid_returns_single_frame_scale_weights():
    B, V, J = 1, 2, 17
    x = torch.rand(B, V, J, 3)
    K = torch.eye(3).repeat(V, 1, 1)
    R = torch.eye(3).repeat(V, 1, 1)
    t = torch.zeros(V, 3)
    t[1, 0] = -1.0
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid(
        j=J,
        d=16,
        n_views=V,
        n_heads=4,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=16,
        principal_point_hidden=8,
        temporal_scales=(1, 0),
        pyramid_layers=1,
        pyramid_n_heads=4,
        return_pp_delta=True,
        return_scale_weights=True,
    ).eval()

    with torch.no_grad():
        pred, weights, pp_delta, scale_weights = model(x, K=K, R=R, t=t)

    assert pred.shape == (B, J, 3)
    assert weights.shape == (B, V, J)
    assert pp_delta.shape == (B, V, 2)
    assert scale_weights.shape == (B, V, J, 2)
