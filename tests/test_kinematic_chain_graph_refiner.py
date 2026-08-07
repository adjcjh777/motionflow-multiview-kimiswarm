"""CPU tests for the kinematic-chain graph refiner and its temporal wrapper."""

import pytest
import torch

from motionflow_mv.fusion.kinematic_chain_graph_refiner import (
    KinematicChainGraphRefiner,
    KinematicChainGraphRefinerTemporal,
)


@pytest.mark.parametrize("j", [17, 28])
def test_kinematic_chain_graph_refiner_shape_and_gradients(j: int) -> None:
    B, J, C = 3, j, 3
    x = torch.randn(B, J, C)
    refiner = KinematicChainGraphRefiner(j=J, hidden_dim=16, num_layers=2)
    out = refiner(x)
    assert out.shape == (B, J, C)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in refiner.parameters())


@pytest.mark.parametrize("j", [17, 28])
def test_temporal_wrapper_shape_and_gradients(j: int) -> None:
    B, T, J, C = 2, 7, j, 3
    x = torch.randn(B, T, J, C)
    refiner = KinematicChainGraphRefinerTemporal(j=J, hidden_dim=16, num_layers=2)
    out = refiner(x)
    assert out.shape == (B, T, J, C)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in refiner.parameters())


def test_temporal_wrapper_disabled_is_identity() -> None:
    B, T, J, C = 2, 5, 17, 3
    x = torch.randn(B, T, J, C)
    refiner = KinematicChainGraphRefinerTemporal(
        j=J, hidden_dim=16, num_layers=2, enabled=False
    )
    out = refiner(x)
    assert torch.equal(out, x)


def test_temporal_wrapper_3d_input() -> None:
    B, J, C = 4, 17, 3
    x = torch.randn(B, J, C)
    refiner = KinematicChainGraphRefinerTemporal(j=J, hidden_dim=16, num_layers=2)
    out = refiner(x)
    assert out.shape == (B, J, C)
