"""Integration smoke tests for OmniMultiViewFusionV5 with v25/v26 + UDP."""

import torch

from motionflow_mv.fusion.omniview_fusion_v5 import OmniMultiViewFusionV5


def _make_cameras(n_views: int = 4):
    import numpy as np
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


def test_v25_udp_forward_backward():
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_multiview_geometry_fusion_v25=True,
        v25_use_geometry_attention=True,
        v25_use_learned_depth_triangulation=True,
        use_uncertainty_depth_proposals_v27=True,
        v27_udp_n_mixtures=1,
    )
    x = torch.rand(2, 3, 4, 17, 3)
    out = model(x, K=K, R=R, t=t)
    pred = out[0]
    assert pred.shape == (2, 3, 17, 3)
    (pred.mean() + out[4]).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_v26_udp_forward_backward():
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_temporal_geometry_fusion_v26=True,
        v26_temporal_window=3,
        use_uncertainty_depth_proposals_v27=True,
        v27_udp_n_mixtures=1,
    )
    x = torch.rand(2, 5, 4, 17, 3)
    out = model(x, K=K, R=R, t=t)
    pred = out[0]
    assert pred.shape == (2, 5, 17, 3)
    (pred.mean() + out[4]).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_v26_udp_gmm_forward_backward():
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_temporal_geometry_fusion_v26=True,
        v26_temporal_window=3,
        use_uncertainty_depth_proposals_v27=True,
        v27_udp_n_mixtures=2,
    )
    x = torch.rand(2, 5, 4, 17, 3)
    out = model(x, K=K, R=R, t=t)
    pred = out[0]
    assert pred.shape == (2, 5, 17, 3)
    (pred.mean() + out[4]).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_v26_udp_gmm_v28_forward_backward():
    K, R, t = _make_cameras(4)
    model = OmniMultiViewFusionV5(
        j=17,
        d=32,
        n_views=4,
        n_st_layers=1,
        graph_num_layers=1,
        use_temporal_geometry_fusion_v26=True,
        v26_temporal_window=3,
        use_uncertainty_depth_proposals_v27=True,
        v27_udp_n_mixtures=2,
        use_physical_space_alignment_v28=True,
        v28_floor_loss_weight=0.01,
        v28_bone_temporal_weight=0.01,
    )
    x = torch.rand(2, 5, 4, 17, 3)
    out = model(x, K=K, R=R, t=t)
    pred = out[0]
    assert pred.shape == (2, 5, 17, 3)
    (pred.mean() + out[4]).backward()
    assert any(p.grad is not None for p in model.parameters())
