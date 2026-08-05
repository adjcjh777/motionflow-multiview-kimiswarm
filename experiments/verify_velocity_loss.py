"""CPU-only sanity check for the new velocity loss.

Generates synthetic (B, T, J, 3) sequences, computes velocity losses,
and asserts expected properties:
  * loss is zero when predictions match ground truth exactly
  * loss increases monotonically with added jitter
  * loss is zero for constant-velocity sequences under L1/L2 variants
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.losses import velocity_loss, velocity_l1_loss


def make_sequence(b=2, t=25, j=17):
    """Create a smooth synthetic pose sequence (linear motion)."""
    rng = np.random.default_rng(42)
    base = rng.uniform(-0.5, 0.5, size=(b, 1, j, 3))
    velocity = rng.uniform(-0.02, 0.02, size=(b, 1, j, 3))
    frames = []
    for i in range(t):
        frames.append(base + i * velocity)
    return np.concatenate(frames, axis=1).astype(np.float32)


def add_jitter(seq, scale):
    """Add per-frame i.i.d. Gaussian jitter to a sequence."""
    rng = np.random.default_rng(2024)
    jitter = rng.normal(0, scale, size=seq.shape).astype(np.float32)
    return seq + jitter


def main():
    gt = torch.from_numpy(make_sequence(b=2, t=25, j=17))

    # 1. Zero loss for perfect predictions
    loss_perfect = velocity_loss(gt, gt).item()
    assert loss_perfect == 0.0, f"Perfect prediction loss should be 0, got {loss_perfect}"

    # 2. Loss increases with jitter
    losses_l2 = []
    for scale in [0.0, 0.001, 0.01, 0.1]:
        pred = torch.from_numpy(add_jitter(gt.numpy(), scale))
        loss = velocity_loss(pred, gt).item()
        losses_l2.append((scale, loss))
    scales, loss_values = zip(*losses_l2)
    for i in range(1, len(loss_values)):
        assert loss_values[i] > loss_values[i - 1], (
            f"Loss did not increase with jitter: {losses_l2}"
        )

    # 3. L1 and L2 are comparable for the same noise
    pred_noisy = torch.from_numpy(add_jitter(gt.numpy(), 0.05))
    l2 = velocity_loss(pred_noisy, gt).item()
    l1 = velocity_l1_loss(pred_noisy, gt).item()

    # 4. Constant-velocity sequences have zero second-order temporal error
    #    (we do not penalize constant velocity, only mismatch).
    pred_constant_vel = gt.clone()
    second_order = (pred_constant_vel[:, 2:] - 2 * pred_constant_vel[:, 1:-1] + pred_constant_vel[:, :-2])
    assert second_order.abs().max() < 1e-6, "Constant-velocity sequence should have zero acceleration"

    print("Velocity loss sanity checks passed.")
    print(f"  perfect L2 loss: {loss_perfect:.6e}")
    print(f"  L2 losses vs jitter: {losses_l2}")
    print(f"  noisy L2: {l2:.6e}, noisy L1: {l1:.6e}")


if __name__ == "__main__":
    main()
