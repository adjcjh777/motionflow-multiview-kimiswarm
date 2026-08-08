import pytest
import torch

from motionflow_mv.fusion.kinematic_anthropometric_prior_v22 import (
    KinematicAnthropometricPrior,
)


def _make_inputs(n=2, t=3, j=17, d=32):
    feat = torch.randn(n * t, j, d)
    pred = torch.randn(n * t, j, 3)
    return feat, pred


@pytest.mark.parametrize("j", [17, 28])
def test_kap_forward_shape_and_loss(j):
    feat, pred = _make_inputs(j=j, d=16)
    model = KinematicAnthropometricPrior(j=j, d=16, hidden=16, residual_hidden=16)
    pred_ref, loss = model(feat, pred)
    assert pred_ref.shape == pred.shape
    assert loss.shape == ()
    assert loss.item() >= 0.0


@pytest.mark.parametrize("j", [17, 28])
def test_kap_backward(j):
    feat, pred = _make_inputs(j=j, d=16)
    feat.requires_grad_(True)
    pred.requires_grad_(True)
    model = KinematicAnthropometricPrior(j=j, d=16, hidden=16, residual_hidden=16)
    pred_ref, loss = model(feat, pred)
    loss.backward()
    assert pred.grad is not None
    # At least one model parameter has a gradient.
    assert any(p.grad is not None for p in model.parameters())


def test_kap_identity_at_init():
    feat, pred = _make_inputs()
    model = KinematicAnthropometricPrior(j=17, d=32)
    pred_ref, _ = model(feat, pred)
    # The residual branch is initialized near zero, so the refined pose should
    # be very close to the input pose.
    assert torch.allclose(pred_ref, pred, atol=1e-1)


def test_kap_residual_magnitude_bounded():
    feat, pred = _make_inputs()
    model = KinematicAnthropometricPrior(j=17, d=32)
    pred_ref, _ = model(feat, pred)
    # Residual branch is initialized near zero, so delta should be within max_delta.
    assert (pred_ref - pred).abs().max().item() <= 0.11


def test_kap_angle_limit_toggle():
    feat, pred = _make_inputs()
    model_on = KinematicAnthropometricPrior(j=17, d=32, use_angle_limit=True)
    model_off = KinematicAnthropometricPrior(j=17, d=32, use_angle_limit=False)
    _, loss_on = model_on(feat, pred)
    _, loss_off = model_off(feat, pred)
    # The loss with angle limit should be greater or equal.
    assert loss_on.item() >= loss_off.item()


def test_kap_unsupported_joints():
    with pytest.raises(ValueError):
        KinematicAnthropometricPrior(j=12, d=16)
