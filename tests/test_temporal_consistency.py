"""CPU smoke test for temporal velocity + acceleration consistency loss."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.losses import (
    TemporalConsistencyLoss,
    acceleration_loss,
    temporal_consistency_loss,
    velocity_loss,
)


def make_sequence(b=2, t=25, j=17):
    """Create a smooth synthetic pose sequence with constant velocity."""
    rng = np.random.default_rng(42)
    base = rng.uniform(-0.5, 0.5, size=(b, 1, j, 3))
    velocity = rng.uniform(-0.02, 0.02, size=(b, 1, j, 3))
    frames = [base + i * velocity for i in range(t)]
    return np.concatenate(frames, axis=1).astype(np.float32)


def test_acceleration_loss_zero_for_perfect_prediction():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    loss = acceleration_loss(pred, gt)
    assert loss.item() == 0.0, f"Perfect prediction acceleration loss should be 0, got {loss.item()}"
    print(f"[OK] acceleration perfect loss: {loss.item():.6e}")


def test_acceleration_loss_zero_for_constant_velocity():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    loss = acceleration_loss(gt, gt)
    assert loss.item() == 0.0, f"Constant-velocity GT should yield zero acceleration loss, got {loss.item()}"
    print(f"[OK] acceleration constant-velocity loss: {loss.item():.6e}")


def test_acceleration_loss_increases_with_jitter():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    losses = []
    for scale in [0.0, 0.001, 0.01, 0.1]:
        rng = np.random.default_rng(2024)
        jitter = rng.normal(0, scale, size=gt.shape).astype(np.float32)
        pred = gt + torch.from_numpy(jitter)
        loss = acceleration_loss(pred, gt).item()
        losses.append((scale, loss))
    for i in range(1, len(losses)):
        assert losses[i][1] > losses[i - 1][1], f"Loss did not increase with jitter: {losses}"
    print(f"[OK] acceleration jitter losses: {losses}")


def test_temporal_consistency_loss_combines_terms():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    loss_zero = temporal_consistency_loss(pred, gt, velocity_weight=1.0, acceleration_weight=1.0)
    assert loss_zero.item() == 0.0, f"Combined loss should be zero for perfect predictions, got {loss_zero.item()}"

    # With only velocity weight, combined loss equals velocity loss
    loss_v = temporal_consistency_loss(pred, gt, velocity_weight=1.0, acceleration_weight=0.0)
    expected_v = velocity_loss(pred, gt)
    assert abs(loss_v.item() - expected_v.item()) < 1e-6, "Velocity-only combined loss mismatch"

    # With only acceleration weight, combined loss equals acceleration loss
    loss_a = temporal_consistency_loss(pred, gt, velocity_weight=0.0, acceleration_weight=1.0)
    expected_a = acceleration_loss(pred, gt)
    assert abs(loss_a.item() - expected_a.item()) < 1e-6, "Acceleration-only combined loss mismatch"
    print("[OK] temporal_consistency_loss combines velocity and acceleration correctly")


def test_temporal_consistency_module():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    module = TemporalConsistencyLoss(velocity_weight=0.5, acceleration_weight=0.3)
    loss = module(pred, gt)
    assert loss.item() == 0.0, f"Module should yield zero loss for perfect predictions, got {loss.item()}"
    print("[OK] TemporalConsistencyLoss module works for perfect predictions")


def test_reduction_modes():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt + torch.randn_like(gt) * 0.01
    mean_loss = acceleration_loss(pred, gt, reduction="mean")
    sum_loss = acceleration_loss(pred, gt, reduction="sum")
    assert mean_loss > 0
    assert sum_loss > 0
    assert isinstance(mean_loss, torch.Tensor)
    assert isinstance(sum_loss, torch.Tensor)
    print("[OK] reduction modes work")


def main():
    test_acceleration_loss_zero_for_perfect_prediction()
    test_acceleration_loss_zero_for_constant_velocity()
    test_acceleration_loss_increases_with_jitter()
    test_temporal_consistency_loss_combines_terms()
    test_temporal_consistency_module()
    test_reduction_modes()
    print("All temporal consistency smoke tests passed.")


if __name__ == "__main__":
    main()
