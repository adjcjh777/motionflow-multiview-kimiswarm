"""Tests for ``SkeletonGraphResidualRefiner`` and its v4 wrapper."""

import pytest
import torch

from motionflow_mv.fusion.skeleton_graph_residual_refiner import (
    SkeletonGraphResidualRefiner,
    SkeletonGraphResidualRefinerWrapper,
)


@pytest.mark.parametrize("j", [17, 28])
def test_skeleton_graph_residual_refiner_forward_shape(j: int) -> None:
    """Input ``(B*T, J, d+3)`` should map to ``(B*T, J, 3)``."""
    d = 64
    in_dim = d + 3
    model = SkeletonGraphResidualRefiner(j=j, in_dim=in_dim, hidden_dim=32, num_layers=2)
    x = torch.randn(4, j, in_dim)
    out = model(x)
    assert out.shape == (4, j, 3)


@pytest.mark.parametrize("j", [17, 28])
def test_skeleton_graph_residual_refiner_wrapper_forward_shape(j: int) -> None:
    """Wrapper should present the same ``(B*T, J, d+3) -> (B*T, J, 3)`` interface."""
    d = 64
    in_dim = d + 3
    model = SkeletonGraphResidualRefinerWrapper(j=j, in_dim=in_dim, hidden_dim=32, num_layers=2)
    x = torch.randn(4, j, in_dim)
    out = model(x)
    assert out.shape == (4, j, 3)


def test_skeleton_graph_residual_refiner_gradients_flow() -> None:
    """A backward pass should populate at least one parameter gradient."""
    model = SkeletonGraphResidualRefiner(j=17, in_dim=67, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 17, 67)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_skeleton_graph_residual_refiner_wrapper_gradients_flow() -> None:
    """Backward pass should flow through the v4 wrapper as well."""
    model = SkeletonGraphResidualRefinerWrapper(j=17, in_dim=67, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 17, 67)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_unsupported_joint_count_raises() -> None:
    """Only H36M 17-joint and MPI-INF-3DHP 28-joint skeletons are supported."""
    with pytest.raises(NotImplementedError):
        SkeletonGraphResidualRefiner(j=25, in_dim=67, hidden_dim=32, num_layers=2)
