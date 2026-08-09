"""Smoke test for v49-Lite lightweight temporal aggregation wiring."""

import pytest
import torch

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


@pytest.mark.parametrize("use_view_count", [True, False])
def test_omniview_v5_with_v49_lite_temporal(use_view_count: bool) -> None:
    """Build OmniMultiViewFusionV5 with v49-Lite temporal head and run a forward pass."""
    B, T, V, J = 2, 9, 4, 17
    model = OmniMultiViewFusionV5(
        j=J,
        n_views=V,
        d=32,
        residual_hidden=64,
        n_st_layers=1,
        graph_num_layers=1,
        n_joint_layers=1,
        n_heads=2,
        use_multiview_geometry_fusion_v25=True,
        use_v46_sparse_view_generalization=True,
        use_v49_lite_temporal_aggregation=True,
        v49_lite_temporal_d_model=16,
        v49_lite_temporal_num_layers=1,
        v49_lite_temporal_use_view_count_conditioning=use_view_count,
    )

    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).expand(B, V, 3, 3)
    R = torch.eye(3).unsqueeze(0).expand(B, V, 3, 3)
    t = torch.zeros(B, V, 3)
    view_mask = torch.ones(B, T, V)

    out = model(x, K=K, R=R, t=t, view_mask=view_mask)
    pred_3d = out[0] if isinstance(out, tuple) else out
    assert pred_3d.shape == (B, T, J, 3)
