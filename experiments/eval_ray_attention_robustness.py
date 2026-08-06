"""Evaluate ray_attention under controlled occlusion and outliers.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/eval_ray_attention_robustness.py \
        --checkpoint outputs/ray_attention_synthetic.pth
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel


def make_cameras(n_views: int = 4, radius: float = 4.0):
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 900.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        c = radius * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        c[2] += 1.5
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def project_points(points_3d: np.ndarray, camera: Camera):
    P = camera.projection_matrix
    X_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    x_h = (P @ X_h.T).T
    return x_h[:, :2] / x_h[:, 2:3]


def synthesize_frame(cameras, rng, noise_std: float = 0.5):
    """Create a random 3D skeleton and project it."""
    J = 17
    # Random skeleton around origin
    joints_3d = rng.normal(0, 0.5, size=(J, 3)) + np.array([0.0, 0.0, 1.0])
    points_2d = np.stack([project_points(joints_3d, cam) for cam in cameras], axis=0)
    points_2d += rng.normal(0, noise_std, size=points_2d.shape)
    confidences = rng.uniform(0.8, 1.0, size=(len(cameras), J))
    return joints_3d, points_2d, confidences


def apply_occlusion(points_2d, confidences, view_indices, rng):
    """Set the given views to zero confidence."""
    for v in view_indices:
        confidences[v] = 0.0
    return points_2d, confidences


def apply_outliers(points_2d, confidences, view_indices, rng):
    """Add large outliers to the given views."""
    for v in view_indices:
        points_2d[v] += rng.normal(0, 50.0, size=points_2d[v].shape)
        confidences[v] = 0.0
    return points_2d, confidences


def mpjpe(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/ray_attention_synthetic.pth")
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RayAttentionFusionModel(j=17, d=64, n_views=args.n_views).to(device)
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    rng = np.random.default_rng(2025)

    # Baseline: clean 4 views
    cameras = make_cameras(args.n_views)
    errors_clean = []
    errors_one_occluded = []
    errors_two_occluded = []
    errors_one_outlier = []

    for _ in range(args.n_trials):
        joints_3d, points_2d, confidences = synthesize_frame(cameras, rng, noise_std=0.8)

        # Clean
        x = torch.from_numpy(np.concatenate([points_2d, confidences[..., None]], axis=-1)).float().to(device)
        with torch.no_grad():
            pred, _ = model(x[None], cameras)
        pred = pred[0].cpu().numpy()
        errors_clean.append(mpjpe(pred, joints_3d))

        # One view occluded
        p2d = points_2d.copy()
        conf = confidences.copy()
        occluded_view = rng.integers(0, args.n_views)
        p2d, conf = apply_occlusion(p2d, conf, [occluded_view], rng)
        x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float().to(device)
        with torch.no_grad():
            pred, _ = model(x[None], cameras)
        pred = pred[0].cpu().numpy()
        errors_one_occluded.append(mpjpe(pred, joints_3d))

        # Two views occluded
        p2d = points_2d.copy()
        conf = confidences.copy()
        occluded_views = rng.choice(args.n_views, size=2, replace=False)
        p2d, conf = apply_occlusion(p2d, conf, occluded_views, rng)
        x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float().to(device)
        with torch.no_grad():
            pred, _ = model(x[None], cameras)
        pred = pred[0].cpu().numpy()
        errors_two_occluded.append(mpjpe(pred, joints_3d))

        # One view as outlier
        p2d = points_2d.copy()
        conf = confidences.copy()
        outlier_view = rng.integers(0, args.n_views)
        p2d, conf = apply_outliers(p2d, conf, [outlier_view], rng)
        x = torch.from_numpy(np.concatenate([p2d, conf[..., None]], axis=-1)).float().to(device)
        with torch.no_grad():
            pred, _ = model(x[None], cameras)
        pred = pred[0].cpu().numpy()
        errors_one_outlier.append(mpjpe(pred, joints_3d))

    print(f"MPJPE (m) over {args.n_trials} trials")
    print(f"  Clean 4 views:        {np.mean(errors_clean):.4f}")
    print(f"  1 view occluded:      {np.mean(errors_one_occluded):.4f}")
    print(f"  2 views occluded:     {np.mean(errors_two_occluded):.4f}")
    print(f"  1 view outlier:       {np.mean(errors_one_outlier):.4f}")


if __name__ == "__main__":
    main()
