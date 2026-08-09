"""Verify the v51 TTSER flag wires into OmniMultiViewFusionV5."""

from __future__ import annotations

import torch

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


def test_v51_tta_flag_wires_into_model() -> None:
    model = OmniMultiViewFusionV5(
        j=17,
        n_views=4,
        d=32,
        n_st_layers=1,
        residual_hidden=64,
        principal_point_hidden=32,
        use_v51_test_time_self_evolution_refiner=True,
        v51_tta_num_steps=1,
    )
    model.eval()

    B, T, V, J = 2, 3, 4, 17
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t)

    pred_3d = out[0]
    assert pred_3d.shape == (B, T, J, 3)
    assert hasattr(model, "v51_tta_last_reliability")
    assert hasattr(model, "v51_tta_last_uncertainty")
    assert model.v51_tta_last_reliability is not None
    assert model.v51_tta_last_uncertainty is not None
    assert model.v51_tta_last_reliability.shape == (B, V)
    assert model.v51_tta_last_uncertainty.shape == (B, J)


def test_v51_tta_disabled_by_default() -> None:
    model = OmniMultiViewFusionV5(
        j=17,
        n_views=4,
        d=32,
        n_st_layers=1,
        residual_hidden=64,
        principal_point_hidden=32,
    )
    model.eval()

    B, T, V, J = 1, 2, 4, 17
    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    R = torch.eye(3)[None, None, :, :].expand(B, V, -1, -1).clone()
    t = torch.randn(B, V, 3)

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t)

    assert len(out) == 5
    assert not model.use_v51_test_time_self_evolution_refiner
