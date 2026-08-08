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
    head.residual_scale.data.fill_(1.0)
    X = torch.randn(2, 3, 17, 3, requires_grad=True)
    out = head(X)
    out.mean().backward()
    assert X.grad is not None


def test_floor_loss_non_negative():
    X = torch.randn(2, 3, 17, 3)
    loss = floor_loss(X, -1.0, [3, 6, 11, 14])
    assert loss >= 0.0
    assert torch.isfinite(loss)


def test_bone_temporal_loss_shape():
    X = torch.randn(2, 5, 17, 3)
    parents = list(range(-1, 16))
    loss = bone_temporal_loss(X, parents)
    assert torch.isfinite(loss)
    assert loss >= 0.0
