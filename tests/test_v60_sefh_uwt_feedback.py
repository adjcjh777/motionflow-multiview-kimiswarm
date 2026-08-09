"""Smoke tests for v60 SEFH -> UWT feedback loop in OmniMultiViewFusionV5."""

import numpy as np
import pytest
import torch

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


def _make_cameras(n_views: int = 4):
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    return (
        torch.from_numpy(np.stack(Ks)).float(),
        torch.from_numpy(np.stack(Rs)).float(),
        torch.from_numpy(np.stack(ts)).float(),
    )


def test_v60_requires_v50_and_v52():
    """v60 should raise when v50 or v52 is missing."""
    with pytest.raises(ValueError, match="use_v60_sefh_uwt_feedback requires"):
        OmniMultiViewFusionV5(
            j=17,
            d=32,
            n_views=4,
            n_heads=4,
            n_st_layers=1,
            graph_num_layers=1,
            use_v60_sefh_uwt_feedback=True,
        )

    with pytest.raises(ValueError, match="use_v60_sefh_uwt_feedback requires"):
        OmniMultiViewFusionV5(
            j=17,
            d=32,
            n_views=4,
            n_heads=4,
            n_st_layers=1,
            graph_num_layers=1,
            use_v50_self_evolution_feedback_head=True,
            use_v60_sefh_uwt_feedback=True,
        )


def test_v60_sefh_uwt_feedback_runs():
    """v60 code path should run and produce a valid 3D pose."""
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_heads=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_full_precision_dlt=True,
        use_camera_view_embedding=True,
        use_v50_self_evolution_feedback_head=True,
        use_v52_uncertainty_weighted_triangulation=True,
        use_v60_sefh_uwt_feedback=True,
    )
    model.eval()

    B, T, V, J = 2, 3, 4, 17
    x = torch.rand(B, T, V, J, 3)
    x[..., 2] = 1.0

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t)
    pred_3d = out[0]
    assert pred_3d.shape == (B, T, J, 3)
    assert not pred_3d.isnan().any()


def test_v60_sefh_uwt_feedback_shapes():
    """v60 should preserve shapes under variable-view masks."""
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_heads=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_full_precision_dlt=True,
        use_camera_view_embedding=True,
        use_v50_self_evolution_feedback_head=True,
        use_v52_uncertainty_weighted_triangulation=True,
        use_v60_sefh_uwt_feedback=True,
    )
    model.eval()

    B, T, V, J = 2, 3, 4, 17
    x = torch.rand(B, T, V, J, 3)
    x[..., 2] = 1.0
    view_mask = torch.ones(B, T, V)
    view_mask[:, :, -1] = 0.0

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t, view_mask=view_mask)
    pred_3d = out[0]
    assert pred_3d.shape == (B, T, J, 3)
    assert not pred_3d.isnan().any()


if __name__ == "__main__":
    test_v60_requires_v50_and_v52()
    test_v60_sefh_uwt_feedback_runs()
    test_v60_sefh_uwt_feedback_shapes()
    print("v60 SEFH -> UWT feedback loop smoke tests passed")
