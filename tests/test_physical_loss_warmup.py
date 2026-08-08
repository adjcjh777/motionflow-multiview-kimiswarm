"""Tests for physical loss warmup."""

import pytest
import torch

from motionflow_mv.fusion.self_evolving_hierarchical_multiview_v29 import PhysicalSpaceTemporalLossV29


def test_physical_loss_warmup_ramps() -> None:
    """Loss scale should increase with epoch during warmup."""
    loss_fn = PhysicalSpaceTemporalLossV29(
        floor_loss_weight=1.0,
        bone_temporal_weight=1.0,
        com_jitter_weight=1.0,
        parents=[0, 0, 1, 2],
        foot_joint_indices=[3, 4],
        warmup_epochs=3,
    )
    X = torch.randn(2, 4, 5, 3)

    loss_fn.set_epoch(0)
    loss0, terms0 = loss_fn(X)

    loss_fn.set_epoch(1)
    loss1, terms1 = loss_fn(X)

    loss_fn.set_epoch(3)
    loss3, terms3 = loss_fn(X)

    assert terms0["scale"].item() == 0.0
    assert terms1["scale"].item() == pytest.approx(1 / 3, abs=1e-5)
    assert terms3["scale"].item() == 1.0
    assert loss0 < loss1 < loss3 or torch.isclose(loss0, torch.tensor(0.0))
