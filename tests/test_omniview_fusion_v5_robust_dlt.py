"""Smoke tests for v8 robust covariance-aware DLT reweighting."""

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


def test_robust_dlt_reweight_runs_with_outlier():
    """A single corrupted view should not crash the robust reweight path."""
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_heads=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_full_precision_dlt=True,
        use_robust_dlt_reweight=True,
        use_camera_view_embedding=True,
        use_set_view_aggregator=True,
    )
    model.eval()

    B, T, V, J = 2, 3, 4, 17
    x = torch.rand(B, T, V, J, 3)
    x[..., 2] = 1.0
    # Corrupt the first view for the last joint to simulate an bad detection.
    x[:, :, 0, -1, :2] += 50.0

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t)
    pred_3d = out[0]
    assert pred_3d.shape == (B, T, J, 3)
    assert not pred_3d.isnan().any()
    assert not out[1].isnan().any()


def test_robust_dlt_reweight_runs_variable_views():
    """Robust reweight should be compatible with variable-view masks."""
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_heads=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_full_precision_dlt=True,
        use_robust_dlt_reweight=True,
        use_camera_view_embedding=True,
    )
    model.eval()

    B, T, V, J = 2, 3, 4, 17
    x = torch.rand(B, T, V, J, 3)
    x[..., 2] = 1.0
    # Mask out the last view.
    view_mask = torch.ones(B, T, V)
    view_mask[:, :, -1] = 0.0

    with torch.no_grad():
        out = model(x, K=K, R=R, t=t, view_mask=view_mask)
    pred_3d = out[0]
    assert pred_3d.shape == (B, T, J, 3)
    assert not pred_3d.isnan().any()


if __name__ == "__main__":
    test_robust_dlt_reweight_runs_with_outlier()
    test_robust_dlt_reweight_runs_variable_views()
    print("v8 robust DLT smoke tests passed")
