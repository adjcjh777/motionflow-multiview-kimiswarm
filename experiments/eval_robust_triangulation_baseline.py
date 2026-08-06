"""Smoke evaluation of the IRLS/Charbonnier robust triangulation baseline.

Compares the new ``robust_triangulation_baseline`` FusionModule against a
torch-only DLT baseline on synthetic multi-view data with and without injected
outliers.

Usage:
    python experiments/eval_robust_triangulation_baseline.py
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.eval.metrics import mpjpe, pa_mpjpe
from motionflow_mv.fusion import FUSION_REGISTRY
from motionflow_mv.fusion.robust_triangulation_baseline_module import (
    RobustTriangulationBaselineFusion,
    register_robust_triangulation_baseline_fusion_module,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


def make_cameras(n_views: int = 5, rng: np.random.Generator = None):
    """Create a deterministic hemispherical multi-camera rig without numpy QR."""
    if rng is None:
        rng = np.random.default_rng(0)
    cameras = []
    for i in range(n_views):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[:2, 2] = rng.uniform(300, 340, size=2)

        theta = 2 * np.pi * i / n_views
        phi = np.pi / 3
        radius = 5.0
        c = radius * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        forward = -c / np.linalg.norm(c)
        temp = np.array([0.0, 0.0, 1.0])
        if abs(forward[2]) > 0.9:
            temp = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, temp)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)
        R = np.vstack([right, up, forward])
        t = -R @ c
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


def _projection_matrices(cameras: List[Camera]) -> torch.Tensor:
    """Return (V, 3, 4) projection matrices computed with PyTorch."""
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0))
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0))
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0))
    Rt = torch.cat([R, t[..., None]], dim=-1)
    return K @ Rt


def generate_sequence(
    n_views: int = 5,
    j: int = 17,
    t: int = 30,
    seed: int = 2025,
    outlier_rate: float = 0.0,
    outlier_std: float = 20.0,
):
    """Generate a synthetic multi-view sequence with optional 2D outliers."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, size=(j, 3)) * np.array([0.5, 0.8, 1.5])
    base[:, 2] += 3.0
    trajectory = np.cumsum(rng.normal(0, 0.05, size=(t, 3)), axis=0)
    joints_3d = base[None, :, :] + trajectory[:, None, :]

    cameras = make_cameras(n_views, rng)
    P = _projection_matrices(cameras)  # (V, 3, 4)

    points_2d = torch.zeros((t, n_views, j, 2), dtype=torch.float64)
    confidences = torch.ones((t, n_views, j), dtype=torch.float64) * 0.9
    X_h = torch.cat(
        [torch.from_numpy(joints_3d), torch.ones((t, j, 1), dtype=torch.float64)],
        dim=-1,
    )  # (T, J, 4)
    for v in range(n_views):
        x_h = (P[v] @ X_h.reshape(-1, 4).T).T.reshape(t, j, 3)
        x = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
        x += torch.from_numpy(rng.normal(0, 0.5, size=(t, j, 2))).to(torch.float64)
        points_2d[:, v] = x

    if outlier_rate > 0:
        mask = torch.from_numpy(rng.random((t, n_views, j)) < outlier_rate)
        offset = torch.from_numpy(rng.normal(0, outlier_std, size=(t, n_views, j, 2))).to(torch.float64)
        points_2d = points_2d + offset * mask[..., None]
        confidences[mask] = 0.1

    return points_2d, confidences, cameras, joints_3d


def dlt_baseline(points_2d: torch.Tensor, confidences: torch.Tensor, cameras: List[Camera]) -> torch.Tensor:
    """Torch-only confidence-weighted DLT baseline (avoids numpy BLAS crashes)."""
    P = _projection_matrices(cameras)
    t, v, j, _ = points_2d.shape
    out = torch.zeros((t, j, 3), dtype=torch.float64)
    for ti in range(t):
        for ji in range(j):
            out[ti, ji] = triangulate_dlt_torch(
                points_2d[ti, :, ji, :],
                P,
                weights=confidences[ti, :, ji],
            )
    return out


def evaluate(n_views: int = 5, outlier_rate: float = 0.0, seed: int = 2025):
    points_2d, confidences, cameras, joints_3d_gt = generate_sequence(
        n_views=n_views, outlier_rate=outlier_rate, seed=seed
    )
    joints_3d_gt_t = torch.from_numpy(joints_3d_gt)

    robust_module = FUSION_REGISTRY.get("robust_triangulation_baseline")

    pred_robust = robust_module.fuse(
        points_2d.numpy(), confidences.numpy(), cameras
    )

    # DLT comparison: use the torch-only implementation because the numpy
    # BLAS/LAPACK stack in this Windows environment raises fatal exceptions.
    pred_dlt = dlt_baseline(points_2d, confidences, cameras)

    return {
        "dlt_mpjpe": mpjpe(pred_dlt.numpy(), joints_3d_gt_t.numpy()),
        "dlt_pa_mpjpe": pa_mpjpe(pred_dlt.numpy(), joints_3d_gt_t.numpy()),
        "robust_mpjpe": mpjpe(pred_robust, joints_3d_gt_t.numpy()),
        "robust_pa_mpjpe": pa_mpjpe(pred_robust, joints_3d_gt_t.numpy()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate IRLS robust triangulation baseline.")
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--outlier_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    register_robust_triangulation_baseline_fusion_module()

    print("Smoke evaluation of IRLS/Charbonnier robust triangulation baseline\n")

    for label, outlier_rate in [("clean", 0.0), ("outlier", args.outlier_rate)]:
        metrics = evaluate(n_views=args.n_views, outlier_rate=outlier_rate, seed=args.seed)
        print(f"Setting: {label} (outlier_rate={outlier_rate:.2f})")
        print(f"  DLT    MPJPE: {metrics['dlt_mpjpe']:.4f}  PA-MPJPE: {metrics['dlt_pa_mpjpe']:.4f}")
        print(f"  IRLS   MPJPE: {metrics['robust_mpjpe']:.4f}  PA-MPJPE: {metrics['robust_pa_mpjpe']:.4f}")
        print()


if __name__ == "__main__":
    main()
