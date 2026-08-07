"""CPU tests for hardened variable-view inference (T11).

Tests the hardening added in ``motionflow_mv/fusion/variable_view_inference.py``:
- explicit view padding/masking with valid camera fills
- graph-joint attention restricted to active views
- confidence-based fallback triangulation when active views < min_views
"""

import numpy as np
import pytest
import torch

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
)
from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)
from motionflow_mv.fusion.variable_view_inference import (
    HardenedVariableViewInferenceWrapper,
    VariableViewInferenceWrapper,
    apply_view_mask,
    build_active_view_edge_index,
    prepare_variable_view_input,
)


def _make_cameras(n_views: int = 4) -> list:
    cameras = []
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
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def test_apply_view_mask_zeros_inactive_views():
    x = torch.rand(2, 4, 17, 3)
    active = torch.tensor([True, True, False, False])
    masked = apply_view_mask(x, active)
    assert torch.allclose(masked[:, active, :, :], x[:, active, :, :])
    assert torch.allclose(masked[:, ~active, :, :], torch.zeros_like(masked[:, ~active, :, :]))


def test_prepare_variable_view_input_fills_camera_params():
    V, J = 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(2, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    x_p, K_p, R_p, t_p, mask = prepare_variable_view_input(
        x, K, R, t, active_views=[0, 1], fill_camera_mode="last_active"
    )
    assert mask.tolist() == [True, True, False, False]
    # Inactive camera slots should be filled with last active view, not zeros.
    assert not torch.allclose(K_p[2], torch.zeros(3, 3))
    assert not torch.allclose(R_p[2], torch.zeros(3, 3))
    assert not torch.allclose(t_p[2], torch.zeros(3))


def test_build_active_view_edge_index_has_correct_nodes():
    J = 17
    mask = torch.tensor([True, True, False, False])
    edge_index, edge_type = build_active_view_edge_index(
        J, H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, mask
    )
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.shape[0]
    assert edge_index.numel() > 0
    # All nodes should belong to active view indices 0 or 1.
    node_ids = edge_index.unique()
    view_ids = (node_ids / J).long()
    assert set(view_ids.tolist()).issubset({0, 1})


def test_variable_view_inference_wrapper_k_views():
    V, J, T = 4, 17, 5
    cameras = _make_cameras(V)
    x = torch.rand(2, T, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    model = RayAttentionFusionModelTemporalResidual(j=J, d=32, n_views=V)
    for k in [2, 3, 4]:
        wrapper = VariableViewInferenceWrapper(model)
        pred, w = wrapper(x, K, R, t, active_views=k)
        assert pred.shape == (2, T, J, 3)
        assert w.shape[-2] == V and w.shape[-1] == J


def test_hardened_variable_view_inference_wrapper_k_views():
    V, J, T = 4, 17, 3
    cameras = _make_cameras(V)
    x = torch.rand(2, T, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    model = OmniMultiViewFusionV2(j=J, d=32, n_views=V, graph_num_layers=1)
    hardened = HardenedVariableViewInferenceWrapper(model, min_views=3)
    for k in [2, 3, 4]:
        pred, weights, visibility, cov, epi = hardened(x, K, R, t, active_views=k)
        assert pred.shape == (2, T, J, 3)
        assert weights.shape == (2, T, V, J)
        assert visibility.shape == (2, T, V, J)
        assert cov.shape == (2, T, V, J, 2, 2)
        assert epi_loss_has_single_element(epi)


def epi_loss_has_single_element(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 1


def test_hardened_fallback_produces_valid_shape_for_single_view():
    """Fallback to DLT cannot triangulate with <2 views, but should not crash."""
    V, J, T = 4, 17, 3
    cameras = _make_cameras(V)
    x = torch.rand(2, T, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    model = OmniMultiViewFusionV2(j=J, d=32, n_views=V, graph_num_layers=1)
    hardened = HardenedVariableViewInferenceWrapper(model, min_views=2)
    # Single active view should still produce a valid shape (zeros as fallback).
    pred, *_ = hardened(x, K, R, t, active_views=[0])
    assert pred.shape == (2, T, J, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
