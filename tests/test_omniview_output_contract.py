"""Output flags for OmniMultiViewFusion prototypes."""

import torch

from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2
from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _make_cameras


def test_omniview_covariance_flag_hides_covariance_slot():
    common = dict(
        j=17,
        d=8,
        n_views=2,
        n_heads=1,
        n_joint_layers=0,
        n_st_layers=0,
        residual_hidden=8,
        principal_point_hidden=8,
        covariance_hidden=8,
        graph_num_layers=0,
        gn_iters=0,
        return_covariance=False,
    )
    models = [
        OmniMultiViewFusionV2(**common),
        OmniMultiViewFusionV3(
            **common,
            use_multiscale_fusion=False,
            use_camera_conditioning=False,
            use_epipolar_bias=False,
        ),
    ]
    x = torch.rand(1, 2, 17, 3)
    cameras = _make_cameras(2)

    for model in models:
        with torch.no_grad():
            output = model.eval()(x, cameras=cameras)
        assert len(output) == 4
        assert output[-1].shape == ()
