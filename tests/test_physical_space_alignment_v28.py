import torch
import pytest

from motionflow_mv.fusion.physical_space_alignment_v28 import (
    PhysicalSpaceAlignmentV28,
    floor_loss,
    bone_temporal_loss,
)


def test_physical_space_alignment_shape():
    head = PhysicalSpaceAlignmentV28(j=17)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert out.shape == (2, 3, 17, 3)
    assert torch.isfinite(out).all()


def test_physical_space_alignment_identity_at_init():
    head = PhysicalSpaceAlignmentV28(j=17)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert torch.allclose(out, X, atol=1e-5)


def test_physical_space_alignment_backward():
    head = PhysicalSpaceAlignmentV28(j=17)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3, requires_grad=True)
    out = head(X)
    out.mean().backward()
    assert X.grad is not None


def test_physical_space_alignment_reg_loss():
    head = PhysicalSpaceAlignmentV28(j=17)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    out, reg_loss = head(X, return_reg_loss=True)
    assert out.shape == X.shape
    assert reg_loss.numel() == 1
    assert reg_loss >= 0.0


def test_physical_space_alignment_residual_bound():
    head = PhysicalSpaceAlignmentV28(j=17, max_residual=0.05)
    head.residual_logit.data.fill_(10.0)
    X = torch.randn(2, 3, 17, 3)
    out = head(X)
    assert (out - X).abs().max().item() <= 0.05 * 1.01


def test_floor_loss_non_negative():
    X = torch.randn(2, 3, 17, 3)
    loss = floor_loss(X, -1.0, [3, 6, 11, 14])
    assert loss >= 0.0
    assert torch.isfinite(loss)


def test_floor_loss_robust_floor_estimate():
    X = torch.zeros(1, 1, 17, 3)
    X[0, 0, [3, 6, 11], 1] = 0.0
    X[0, 0, 14, 1] = -1.0
    loss = floor_loss(X, 0.0, [3, 6, 11, 14], floor_quantile=0.25)
    assert loss >= 0.0
    assert loss < 0.5


def test_bone_temporal_loss_shape():
    X = torch.randn(2, 5, 17, 3)
    parents = list(range(-1, 16))
    loss = bone_temporal_loss(X, parents)
    assert torch.isfinite(loss)
    assert loss >= 0.0
