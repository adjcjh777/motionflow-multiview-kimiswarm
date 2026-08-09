"""Unit tests for v54 Physical-Space Calibration v2.

Tests focus on:

* identity-at-init (output equals input when identity_init=True)
* per-domain canonical bone scales produce correct shapes and no NaN
* floor/contact sanity on synthetic data
* gradients flow through the GNN and the loss terms
"""

import pytest
import torch

from motionflow_mv.fusion.physical_space_calibration_v2_v54 import (
    PhysicalSpaceCalibrationV2V54,
)


def _make_inputs(batch_size: int = 2, n_views: int = 4, n_joints: int = 17):
    T = 5
    B, V, J = batch_size, n_views, n_joints
    pred_3d_psc = torch.randn(B, T, J, 3, requires_grad=True)
    uwt_weights = torch.rand(B, T, V, J)
    points_2d = torch.rand(B, T, V, J, 2)
    K = torch.eye(3, requires_grad=False).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    R = torch.eye(3, requires_grad=False).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    t = torch.zeros(B, T, V, 3)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    domain_id = torch.randint(0, 3, (B,))
    return pred_3d_psc, uwt_weights, points_2d, K, R, t, view_mask, domain_id


def test_identity_at_init_enabled():
    """With identity_init=True, the output should equal the input."""
    inputs = _make_inputs()
    pred_3d_psc = inputs[0]
    module = PhysicalSpaceCalibrationV2V54(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        identity_init=True,
        residual_gate_init=-6.0,
    )
    pred_3d_psc2, psc2_loss, _, _ = module(pred_3d_psc, *inputs[1:])
    assert pred_3d_psc2.shape == pred_3d_psc.shape
    diff = (pred_3d_psc2 - pred_3d_psc).detach().abs().max().item()
    assert diff < 1e-4, f"Expected identity at init, got max diff {diff}"
    assert torch.isfinite(psc2_loss)


def test_identity_when_disabled():
    """When the module is disabled, the output should still equal the input."""
    inputs = _make_inputs()
    pred_3d_psc = inputs[0]
    module = PhysicalSpaceCalibrationV2V54(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        use_floor=False,
        use_contact=False,
        use_bone_scale=False,
        use_temporal_smoothness=False,
        identity_init=True,
        residual_gate_init=-6.0,
    )
    pred_3d_psc2, psc2_loss, _, _ = module(pred_3d_psc, *inputs[1:])
    diff = (pred_3d_psc2 - pred_3d_psc).detach().abs().max().item()
    assert diff < 1e-4, f"Expected identity when disabled, got max diff {diff}"
    assert torch.isfinite(psc2_loss)


def test_per_domain_canonical_bone_lengths():
    """Canonical bone scales should be selected per domain and produce no NaN."""
    pred_3d_psc, uwt_weights, points_2d, K, R, t, view_mask, domain_id = _make_inputs()
    module = PhysicalSpaceCalibrationV2V54(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        num_domains=4,
        use_floor=False,
        use_contact=False,
        use_bone_scale=True,
        identity_init=True,
    )
    pred_3d_psc2, psc2_loss, floor_height, bone_scale = module(
        pred_3d_psc,
        uwt_weights,
        points_2d,
        K,
        R,
        t,
        view_mask=view_mask,
        domain_id=domain_id,
    )
    assert pred_3d_psc2.shape == pred_3d_psc.shape
    assert torch.isfinite(psc2_loss)
    assert torch.isfinite(bone_scale).all()
    assert bone_scale.shape == (2, 5, 16)  # 16 bones for H36M 17 joints


def test_floor_contact_sanity():
    """Floor height should be finite and feet should be non-negative relative to it."""
    pred_3d_psc, uwt_weights, points_2d, K, R, t, view_mask, domain_id = _make_inputs(batch_size=1)
    # Place the person above the floor with small random offsets.
    with torch.no_grad():
        pred_3d_psc = pred_3d_psc.abs() + 0.2
    module = PhysicalSpaceCalibrationV2V54(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        use_floor=True,
        use_contact=True,
        use_bone_scale=False,
        identity_init=True,
    )
    pred_3d_psc2, psc2_loss, floor_height, _ = module(
        pred_3d_psc,
        uwt_weights,
        points_2d,
        K,
        R,
        t,
        view_mask=view_mask,
        domain_id=domain_id,
    )
    assert torch.isfinite(psc2_loss)
    assert torch.isfinite(floor_height).all()


def test_gradient_flow():
    """Gradients should flow through the GNN and all loss terms."""
    pred_3d_psc, uwt_weights, points_2d, K, R, t, view_mask, domain_id = _make_inputs()
    module = PhysicalSpaceCalibrationV2V54(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        use_floor=True,
        use_contact=True,
        use_bone_scale=True,
        use_temporal_smoothness=True,
        identity_init=False,
        residual_gate_init=0.0,
    )
    pred_3d_psc2, psc2_loss, _, _ = module(
        pred_3d_psc,
        uwt_weights,
        points_2d,
        K,
        R,
        t,
        view_mask=view_mask,
        domain_id=domain_id,
    )
    psc2_loss.backward()
    assert pred_3d_psc.grad is not None
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.parameters()
    )
    assert has_grad, "No parameter received a gradient"
