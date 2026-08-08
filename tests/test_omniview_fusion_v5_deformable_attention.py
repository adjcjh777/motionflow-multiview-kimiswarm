"""Smoke tests for v18 deformable cross-view attention integration into OmniMultiViewFusionV5."""

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


def test_deformable_attention_v18_runs():
    """v18 deformable cross-view attention should produce valid 3D pose."""
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
        use_deformable_cross_view_attention_v18=True,
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


def test_deformable_attention_v18_topk_straight_through():
    """v18 with straight-through top-k sampling should produce valid 3D pose."""
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
        use_deformable_cross_view_attention_v18=True,
        deformable_attention_use_topk_st=True,
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


def test_deformable_attention_v18_variable_views():
    """v18 should respect variable-view masks."""
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
        use_deformable_cross_view_attention_v18=True,
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
    test_deformable_attention_v18_runs()
    test_deformable_attention_v18_variable_views()
    print("v18 deformable cross-view attention integration smoke tests passed")
