"""CPU smoke test for temporal velocity + acceleration consistency loss (v2)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.losses import (
    TemporalConsistencyLossV2,
    acceleration_loss_v2,
    temporal_consistency_loss_v2,
    velocity_loss_v2,
)


def make_sequence(b=2, t=25, j=17):
    """Create a smooth synthetic pose sequence with constant velocity."""
    rng = np.random.default_rng(42)
    base = rng.uniform(-0.5, 0.5, size=(b, 1, j, 3))
    velocity = rng.uniform(-0.02, 0.02, size=(b, 1, j, 3))
    frames = [base + i * velocity for i in range(t)]
    return np.concatenate(frames, axis=1).astype(np.float32)


def test_velocity_loss_v2_zero_for_perfect_prediction():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    loss = velocity_loss_v2(pred, gt)
    assert loss.item() == 0.0, f"Perfect prediction velocity loss should be 0, got {loss.item()}"
    print(f"[OK] v2 velocity perfect loss: {loss.item():.6e}")


def test_velocity_loss_v2_short_sequence():
    gt = torch.from_numpy(make_sequence(b=2, t=1, j=17))
    pred = gt.clone()
    loss = velocity_loss_v2(pred, gt)
    assert loss.item() == 0.0, "Single-frame sequence should yield zero velocity loss"
    print("[OK] v2 velocity short-sequence handling")


def test_acceleration_loss_v2_zero_for_constant_velocity():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    loss = acceleration_loss_v2(gt, gt)
    assert loss.item() == 0.0, f"Constant-velocity GT should yield zero acceleration loss, got {loss.item()}"
    print(f"[OK] v2 acceleration constant-velocity loss: {loss.item():.6e}")


def test_acceleration_loss_v2_increases_with_jitter():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    losses = []
    for scale in [0.0, 0.001, 0.01, 0.1]:
        rng = np.random.default_rng(2024)
        jitter = rng.normal(0, scale, size=gt.shape).astype(np.float32)
        pred = gt + torch.from_numpy(jitter)
        loss = acceleration_loss_v2(pred, gt).item()
        losses.append((scale, loss))
    for i in range(1, len(losses)):
        assert losses[i][1] > losses[i - 1][1], f"Loss did not increase with jitter: {losses}"
    print(f"[OK] v2 acceleration jitter losses: {losses}")


def test_temporal_consistency_loss_v2_combines_terms():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    loss_zero = temporal_consistency_loss_v2(
        pred, gt, velocity_weight=1.0, acceleration_weight=1.0
    )
    assert loss_zero.item() == 0.0, f"Combined loss should be zero for perfect predictions, got {loss_zero.item()}"

    loss_v = temporal_consistency_loss_v2(pred, gt, velocity_weight=1.0, acceleration_weight=0.0)
    expected_v = velocity_loss_v2(pred, gt)
    assert abs(loss_v.item() - expected_v.item()) < 1e-6, "Velocity-only combined loss mismatch"

    loss_a = temporal_consistency_loss_v2(pred, gt, velocity_weight=0.0, acceleration_weight=1.0)
    expected_a = acceleration_loss_v2(pred, gt)
    assert abs(loss_a.item() - expected_a.item()) < 1e-6, "Acceleration-only combined loss mismatch"
    print("[OK] v2 temporal_consistency_loss combines velocity and acceleration correctly")


def test_temporal_consistency_loss_v2_module():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt.clone()
    module = TemporalConsistencyLossV2(velocity_weight=0.5, acceleration_weight=0.3)
    loss = module(pred, gt)
    assert loss.item() == 0.0, f"Module should yield zero loss for perfect predictions, got {loss.item()}"
    print("[OK] TemporalConsistencyLossV2 module works for perfect predictions")


def test_huber_loss_smaller_than_l2_for_outliers():
    rng = np.random.default_rng(99)
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt + torch.from_numpy(rng.normal(0, 0.5, size=gt.shape).astype(np.float32))
    l2 = temporal_consistency_loss_v2(pred, gt, loss_type="l2")
    huber = temporal_consistency_loss_v2(pred, gt, loss_type="huber", delta=0.1)
    assert huber.item() < l2.item(), "Huber loss should be smaller than L2 for large outliers"
    print("[OK] v2 Huber robustness reduces outlier penalty")


def test_masking_ignores_invalid_frames():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    # Add large noise only to frames we will mark as invalid.
    noise = torch.zeros_like(gt)
    noise[:, 10:15, :, :] = 100.0
    pred = gt + noise

    mask = torch.ones_like(gt[:, :, :, 0])  # (B, T, J)
    mask[:, 10:15, :] = 0.0

    loss_masked = velocity_loss_v2(pred, gt, mask=mask)
    loss_unmasked = velocity_loss_v2(pred, gt)
    # Masked loss must ignore the huge invalid-region jitter.
    assert loss_masked.item() < loss_unmasked.item() / 1e4, (
        f"Masked loss {loss_masked.item()} should be much smaller than unmasked {loss_unmasked.item()}"
    )
    print(f"[OK] v2 masking ignores invalid frames: masked={loss_masked.item():.6e}, unmasked={loss_unmasked.item():.6e}")


def test_per_joint_weights():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt + torch.randn_like(gt) * 0.01
    joint_weights = torch.ones(17)
    joint_weights[0] = 0.0
    loss_zero_joint = velocity_loss_v2(pred, gt, joint_weights=joint_weights)
    # Loss with first joint weighted to zero should be smaller than uniform weighting.
    loss_uniform = velocity_loss_v2(pred, gt)
    assert loss_zero_joint.item() < loss_uniform.item(), "Zero-weighted joint should reduce loss"
    print("[OK] v2 per-joint weights affect loss")


def test_reduction_modes():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred = gt + torch.randn_like(gt) * 0.01
    mean_loss = acceleration_loss_v2(pred, gt, reduction="mean")
    sum_loss = acceleration_loss_v2(pred, gt, reduction="sum")
    none_loss = acceleration_loss_v2(pred, gt, reduction="none")
    assert mean_loss > 0
    assert sum_loss > 0
    assert none_loss.dim() == 3  # (B, T-2, J)
    print("[OK] v2 reduction modes work")


def test_gradient_flow():
    pred = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    pred.requires_grad_(True)
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))
    module = TemporalConsistencyLossV2(velocity_weight=1.0, acceleration_weight=1.0)
    loss = module(pred, gt)
    loss.backward()
    assert pred.grad is not None, "Gradient should flow through predictions"
    assert not torch.isnan(pred.grad).any(), "Gradient contains NaN"
    print("[OK] v2 gradient flow works")


def main():
    test_velocity_loss_v2_zero_for_perfect_prediction()
    test_velocity_loss_v2_short_sequence()
    test_acceleration_loss_v2_zero_for_constant_velocity()
    test_acceleration_loss_v2_increases_with_jitter()
    test_temporal_consistency_loss_v2_combines_terms()
    test_temporal_consistency_loss_v2_module()
    test_huber_loss_smaller_than_l2_for_outliers()
    test_masking_ignores_invalid_frames()
    test_per_joint_weights()
    test_reduction_modes()
    test_gradient_flow()
    print("All v2 temporal consistency smoke tests passed.")


if __name__ == "__main__":
    main()
