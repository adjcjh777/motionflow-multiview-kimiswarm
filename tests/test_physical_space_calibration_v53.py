"""Unit tests for v53 Physical-Space Calibration.

Tests focus on:

* identity-at-init (output equals input when disabled or identity_init=True)
* per-domain canonical bone lengths produce correct shapes and no NaN
* gradients flow through the residual MLP and the three loss terms
"""

import pytest
import torch

from motionflow_mv.fusion.physical_space_calibration_v53 import (
    PhysicalSpaceCalibrationV53,
)


def _make_inputs(batch_size: int = 2, n_views: int = 4, n_joints: int = 17):
    T = 3
    B, V, J = batch_size, n_views, n_joints
    pred_3d_uwt = torch.randn(B, T, J, 3, requires_grad=True)
    uwt_weights = torch.rand(B, T, V, J)
    points_2d = torch.rand(B, T, V, J, 2)
    K = torch.eye(3, requires_grad=False).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    R = torch.eye(3, requires_grad=False).view(1, 1, 1, 3, 3).expand(B, T, V, -1, -1).clone()
    t = torch.zeros(B, T, V, 3)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    domain_id = torch.randint(0, 3, (B,))
    return pred_3d_uwt, uwt_weights, points_2d, K, R, t, view_mask, domain_id


def test_identity_at_init_enabled():
    """With identity_init=True, the output should equal the input."""
    inputs = _make_inputs()
    pred_3d_uwt = inputs[0]
    module = PhysicalSpaceCalibrationV53(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        identity_init=True,
        residual_gate_init=-6.0,
    )
    pred_3d_psc, psc_loss, _, _ = module(pred_3d_uwt, *inputs[1:])
    assert pred_3d_psc.shape == pred_3d_uwt.shape
    diff = (pred_3d_psc - pred_3d_uwt).detach().abs().max().item()
    assert diff < 1e-4, f"Expected identity at init, got max diff {diff}"
    assert torch.isfinite(psc_loss)


def test_identity_when_disabled():
    """When the module is disabled (use_floor=False, use_bone_scale=False),
    the output should still equal the input because the residual is gated."""
    inputs = _make_inputs()
    pred_3d_uwt = inputs[0]
    module = PhysicalSpaceCalibrationV53(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        use_floor=False,
        use_bone_scale=False,
        identity_init=True,
        residual_gate_init=-6.0,
    )
    pred_3d_psc, psc_loss, _, _ = module(pred_3d_uwt, *inputs[1:])
    diff = (pred_3d_psc - pred_3d_uwt).detach().abs().max().item()
    assert diff < 1e-4, f"Expected identity when disabled, got max diff {diff}"
    # The loss should still be finite (reproj term is active by default).
    assert torch.isfinite(psc_loss)


def test_per_domain_canonical_bone_lengths():
    """Canonical bone lengths should be selected per domain and produce no NaN."""
    pred_3d_uwt, uwt_weights, points_2d, K, R, t, view_mask, domain_id = _make_inputs()
    module = PhysicalSpaceCalibrationV53(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        num_domains=4,
        use_floor=False,
        use_bone_scale=True,
        identity_init=True,
    )
    pred_3d_psc, psc_loss, floor_height, bone_scale = module(
        pred_3d_uwt,
        uwt_weights,
        points_2d,
        K,
        R,
        t,
        view_mask=view_mask,
        domain_id=domain_id,
    )
    assert pred_3d_psc.shape == pred_3d_uwt.shape
    assert torch.isfinite(psc_loss)
    assert torch.isfinite(bone_scale).all()
    assert bone_scale.shape == (2, 3, 16)  # 16 bones for H36M 17 joints


def test_gradient_flow():
    """Gradients should flow through the residual MLP and all three loss terms."""
    pred_3d_uwt, uwt_weights, points_2d, K, R, t, view_mask, domain_id = _make_inputs()
    module = PhysicalSpaceCalibrationV53(
        j=17,
        n_views=4,
        hidden=64,
        n_layers=2,
        use_floor=True,
        use_bone_scale=True,
        identity_init=False,
        residual_gate_init=0.0,
    )
    pred_3d_psc, psc_loss, _, _ = module(
        pred_3d_uwt,
        uwt_weights,
        points_2d,
        K,
        R,
        t,
        view_mask=view_mask,
        domain_id=domain_id,
    )
    psc_loss.backward()
    assert pred_3d_uwt.grad is not None
    # At least one parameter of the residual MLP has a non-zero gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.parameters()
    )
    assert has_grad, "No parameter received a gradient"
