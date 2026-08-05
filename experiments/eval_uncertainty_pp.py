"""CPU smoke test for the cross-view residual + principal-point + uncertainty model.

Usage
-----
    python experiments/eval_uncertainty_pp.py --n_views 6 --j 17 --clip_len 9

The 2D observations and 3D ground truth are synthetic and geometrically
meaningless; this script only verifies that the model builds, runs a forward
and backward pass on CPU, and produces the expected output shapes/statistics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.models.crossview_residual_uncertainty import (
    CrossviewResidualUncertaintyModel,
)


def _make_synthetic_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras."""
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / (np.linalg.norm(c) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _make_synthetic_data(n_views: int, j: int, n_frames: int):
    """Return synthetic observations and a dummy 3D skeleton."""
    rng = np.random.default_rng(2026)
    cameras = _make_synthetic_cameras(n_views)
    points_2d = rng.uniform(0, 1, size=(n_frames, n_views, j, 2)).astype(np.float32)
    confidences = rng.uniform(0.5, 1.0, size=(n_frames, n_views, j)).astype(np.float32)
    joints_3d = rng.uniform(-1, 1, size=(n_frames, j, 3)).astype(np.float32)
    return points_2d, confidences, joints_3d, cameras


def main():
    parser = argparse.ArgumentParser(description="CPU smoke test for CrossviewResidualUncertaintyModel")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    points_2d, confidences, joints_3d, cameras = _make_synthetic_data(
        args.n_views, args.j, args.clip_len
    )

    # Build model and run on CPU.
    model = CrossviewResidualUncertaintyModel(
        j=args.j,
        d=args.d,
        n_views=args.n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        uncertainty_loss_weight=args.uncertainty_loss_weight,
    )
    model.eval()

    x = torch.from_numpy(np.concatenate([points_2d, confidences[..., None]], axis=-1)).float()
    # Add batch dimension: (B=1, T, V, J, 3)
    x = x.unsqueeze(0)

    print("Running CPU smoke test for CrossviewResidualUncertaintyModel")
    print(f"  input shape: {x.shape}")
    print(f"  n_params: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        pred_3d, weights, log_var, nll_loss = model(x, cameras=cameras)

    print(f"  pred_3d shape: {pred_3d.shape}")
    print(f"  weights shape: {weights.shape}")
    print(f"  log_var shape: {log_var.shape}")
    print(f"  nll_loss: {nll_loss.item():.6f}")
    print(f"  log_var mean: {log_var.mean().item():.4f}, std: {log_var.std().item():.4f}")
    print(f"  weights mean: {weights.mean().item():.4f}, std: {weights.std().item():.4f}")

    # Verify backward pass works.
    x_grad = x.clone().requires_grad_(True)
    pred_3d, weights, log_var, nll_loss = model(x_grad, cameras=cameras)
    loss = pred_3d.mean() + nll_loss
    loss.backward()
    assert x_grad.grad is not None, "Input gradient not computed"
    print(f"  backward pass OK (loss={loss.item():.6f})")

    # Dummy metric against synthetic ground truth.
    gt = torch.from_numpy(joints_3d).float().unsqueeze(0)
    with torch.no_grad():
        err = (pred_3d - gt).norm(dim=-1).mean().item()
    print(f"  dummy MPJPE vs random GT: {err:.4f} m ({err * 1000:.2f} mm)")

    print("CPU smoke test passed.")


if __name__ == "__main__":
    main()
