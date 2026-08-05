"""CPU sanity check: uncertainty head down-weights a deliberately noisy view.

This extends the synthetic smoke test for ``CrossviewResidualUncertaintyModel`` by
injecting a controlled 2-D offset into one camera view and checking that the
predicted log-variance for that view is larger (higher uncertainty / lower
precision) than for the clean views.

Usage
-----
    python experiments/eval_uncertainty_pp_view_robustness.py

Exit code 0 only if the noisy view receives strictly higher per-joint log-variance
and lower mean precision than the clean views.
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
    parser = argparse.ArgumentParser(
        description="CPU sanity check that uncertainty head down-weights a noisy view"
    )
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.1)
    parser.add_argument("--noisy_view", type=int, default=1, help="Index of the view to perturb")
    parser.add_argument("--noise_px", type=float, default=50.0, help="Pixel offset added to the noisy view")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    points_2d, confidences, joints_3d, cameras = _make_synthetic_data(
        args.n_views, args.j, args.clip_len
    )

    # Perturb one view.
    points_2d_noisy = points_2d.copy()
    points_2d_noisy[:, args.noisy_view, :, 0] += args.noise_px

    model = CrossviewResidualUncertaintyModel(
        j=args.j,
        d=args.d,
        n_views=args.n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        uncertainty_loss_weight=args.uncertainty_loss_weight,
    )
    model.eval()

    x = torch.from_numpy(
        np.concatenate([points_2d_noisy, confidences[..., None]], axis=-1)
    ).float().unsqueeze(0)

    with torch.no_grad():
        pred_3d, weights, log_var, nll_loss = model(x, cameras=cameras)

    # log_var shape: (B=1, T, V, J) after squeeze it is (T, V, J) because squeeze_output=True
    # Actually model returns (B, T, V, J) when input is (B, T, V, J, 3). Here input is (1, T, V, J, 3)
    # so pred_3d is (1, T, J, 3), weights/log_var are (1, T, V, J).
    log_var = log_var[0]  # (T, V, J)
    weights = weights[0]  # (T, V, J)

    # Mean over time and joints for each view.
    mean_log_var = log_var.mean(dim=(0, 2))  # (V,)
    mean_precision = torch.exp(-log_var).mean(dim=(0, 2))  # (V,)

    print("Per-view mean log-variance and precision:")
    for v in range(args.n_views):
        marker = " <-- noisy view" if v == args.noisy_view else ""
        print(
            f"  view {v}: log_var={mean_log_var[v].item():.4f}, "
            f"precision={mean_precision[v].item():.4f}{marker}"
        )

    noisy_log_var = mean_log_var[args.noisy_view].item()
    noisy_precision = mean_precision[args.noisy_view].item()
    clean_log_var = torch.cat([
        mean_log_var[:args.noisy_view],
        mean_log_var[args.noisy_view + 1:],
    ], dim=0)
    clean_precision = torch.cat([
        mean_precision[:args.noisy_view],
        mean_precision[args.noisy_view + 1:],
    ], dim=0)

    print()
    print(f"Noisy view log_var: {noisy_log_var:.4f}")
    print(f"Clean views log_var mean: {clean_log_var.mean().item():.4f}  max: {clean_log_var.max().item():.4f}")
    print(f"Noisy view precision: {noisy_precision:.4f}")
    print(f"Clean views precision mean: {clean_precision.mean().item():.4f}  min: {clean_precision.min().item():.4f}")

    success = (
        noisy_log_var > clean_log_var.max().item()
        and noisy_precision < clean_precision.min().item()
    )

    if success:
        print("PASS: noisy view is correctly assigned higher uncertainty and lower precision.")
    else:
        print("FAIL: noisy view did not receive the highest uncertainty / lowest precision.")

    print(f"nll_loss: {nll_loss.item():.6f}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
