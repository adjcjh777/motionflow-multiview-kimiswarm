"""Unit tests for the per-view attention-entropy regularisation loss."""

import pytest
import torch

from motionflow_mv.fusion.attention_entropy_loss import AttentionEntropyLoss


@pytest.fixture
def weights_4d():
    B, T, V, J = 2, 5, 4, 17
    return torch.rand(B, T, V, J, dtype=torch.float32)


def test_entropy_loss_is_non_negative(weights_4d):
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    loss = loss_fn(weights_4d)
    assert loss.numel() == 1
    assert loss.item() >= 0.0


def test_entropy_loss_zero_for_one_hot():
    B, T, V, J = 2, 5, 4, 17
    one_hot = torch.zeros(B, T, V, J, dtype=torch.float32)
    one_hot[:, :, 0, :] = 1.0
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    loss = loss_fn(one_hot)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_loss_disabled_when_weight_is_zero(weights_4d):
    loss_fn = AttentionEntropyLoss(weight=0.0, dim=-2)
    loss = loss_fn(weights_4d)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_entropy_loss_gradient_flow(weights_4d):
    weights = weights_4d.clone().requires_grad_(True)
    loss_fn = AttentionEntropyLoss(weight=0.01, dim=-2)
    loss = loss_fn(weights)
    loss.backward()
    assert weights.grad is not None
    assert weights.grad.shape == weights.shape
    assert not torch.any(torch.isnan(weights.grad))


def test_entropy_loss_3d_input():
    B, V, J = 3, 4, 17
    weights = torch.rand(B, V, J, dtype=torch.float32)
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    loss = loss_fn(weights)
    assert loss.numel() == 1
    assert loss.item() >= 0.0


def test_entropy_loss_uniform_is_maximum():
    B, V, J = 2, 4, 17
    uniform = torch.ones(B, V, J, dtype=torch.float32)
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    loss_uniform = loss_fn(uniform).item()

    # Compare against a slightly peaked distribution; entropy should be lower.
    peaked = torch.ones(B, V, J, dtype=torch.float32)
    peaked[:, 0, :] = 2.0
    loss_peaked = loss_fn(peaked).item()

    assert loss_uniform > loss_peaked


def test_entropy_loss_invalid_reduction():
    with pytest.raises(ValueError):
        AttentionEntropyLoss(reduction="max")


def test_entropy_loss_nans_raise(weights_4d):
    weights = weights_4d.clone()
    weights[0, 0, 0, 0] = float("nan")
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    with pytest.raises(ValueError):
        loss_fn(weights)


def test_entropy_loss_negative_weights_raise():
    weights = torch.ones(2, 4, 17)
    weights[0, 0, 0] = -1.0
    loss_fn = AttentionEntropyLoss(weight=1.0, dim=-2)
    with pytest.raises(ValueError):
        loss_fn(weights)
