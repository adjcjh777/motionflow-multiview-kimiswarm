"""CPU-only diagnostic for camera-calibration robustness.

Projects a synthetic 3D skeleton into a small multi-view rig, perturbs the
calibration in realistic ways (rotation, translation, focal length, principal
point, radial distortion), and triangulates with weighted DLT.  The resulting
MPJPE matrix tells us how much each calibration error type currently hurts the
baseline triangulator and where the next curriculum should focus.

This script does not load any trained model and requires no GPU, so it is safe
to run while the RTX 4090 is training another job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.calibration.perturb import perturb_extrinsics, perturb_intrinsics_with_delta
from motionflow_mv.fusion.ray_attention_model import _triangulate_weighted_dlt


SYNTH_JOINTS_17 = np.array([
    [0.000, 0.000, 0.000],
    [0.000, 0.200, 0.000],
    [0.000, 0.500, 0.000],
    [-0.200, 0.750, 0.050],
    [0.200, 0.750, -0.050],
    [-0.350, 0.500, 0.100],
    [0.350, 0.500, -0.100],
    [-0.450, 0.250, 0.120],
    [0.450, 0.250, -0.120],
    [-0.300, 0.000, 0.080],
    [0.300, 0.000, -0.080],
    [-0.250, -0.400, 0.050],
    [0.250, -0.400, -0.050],
    [-0.200, -0.800, 0.020],
    [0.200, -0.800, -0.020],
    [-0.150, -1.100, 0.000],
    [0.150, -1.100, 0.000],
], dtype=np.float32)


def make_ring_cameras(n_views: int = 4, radius: float = 5.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (K, R, t) for n_views cameras looking at the origin."""
    Ks, Rs, ts = [], [], []
    for i in range(n_views):
        theta = 2.0 * np.pi * i / n_views
        center = np.array([radius * np.cos(theta), radius * np.sin(theta), 1.2], dtype=np.float32)
        z_axis = -center / (np.linalg.norm(center) + 1e-8)
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(tmp, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
        K = np.array([[800.0, 0.0, 320.0],
                      [0.0, 800.0, 240.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        Ks.append(K)
        Rs.append(R)
        ts.append(R @ (-center))
    return torch.from_numpy(np.stack(Ks)), torch.from_numpy(np.stack(Rs)), torch.from_numpy(np.stack(ts))


def project(points_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project (B, J, 3) points with (V, 3, 3)/(V, 3, 3)/(V, 3) cameras."""
    B, J, _ = points_3d.shape
    V = K.shape[0]
    # X_cam = R X + t
    X_cam = torch.einsum("vij,bnj->bvni", R, points_3d) + t.view(1, V, 1, 3)
    x = X_cam[..., 0] / (X_cam[..., 2] + 1e-8)
    y = X_cam[..., 1] / (X_cam[..., 2] + 1e-8)
    xy1 = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (B, V, J, 3)
    xy = torch.einsum("vik,bvjk->bvji", K, xy1)[..., :2]
    return xy


def apply_radial_distortion(xy: torch.Tensor, K: torch.Tensor, k1: float) -> torch.Tensor:
    """Apply radial distortion in normalized image coordinates.

    x_n = (x - cx) / fx, y_n = (y - cy) / fy
    r^2 = x_n^2 + y_n^2
    x_dist = x + (x - cx) * k1 * r^2
    y_dist = y + (y - cy) * k1 * r^2
    """
    if abs(k1) < 1e-9:
        return xy
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    while cx.dim() < xy.dim() - 1:
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
    x = xy[..., 0]
    y = xy[..., 1]
    x_n = (x - cx) / fx
    y_n = (y - cy) / fy
    r2 = x_n * x_n + y_n * y_n
    xy_out = xy.clone()
    xy_out[..., 0] = x + (x - cx) * (k1 * r2)
    xy_out[..., 1] = y + (y - cy) * (k1 * r2)
    return xy_out


def triangulate_and_evaluate(points_2d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                             points_3d_gt: torch.Tensor) -> float:
    """Triangulate with equal weights and return MPJPE in millimeters."""
    B, V, J, _ = points_2d.shape
    weights = torch.ones(B, V, J)
    Rt = torch.cat([R, t[..., None]], dim=-1).expand(B, -1, -1, -1)
    P = K @ Rt
    pred = _triangulate_weighted_dlt(points_2d, weights, P)
    # root-relative for fair comparison with pose literature
    pred_rel = pred - pred[:, 0:1, :]
    gt_rel = points_3d_gt - points_3d_gt[:, 0:1, :]
    return float(torch.norm(pred_rel - gt_rel, dim=-1).mean().item()) * 1000.0


def run_diagnostic(seed: int = 42, n_views: int = 4, noise_px: float = 1.0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    K, R, t = make_ring_cameras(n_views=n_views)
    K = K.unsqueeze(0)  # (1, V, 3, 3)
    R = R.unsqueeze(0)
    t = t.unsqueeze(0)

    points_3d = torch.from_numpy(SYNTH_JOINTS_17[None, ...].copy())
    points_2d_clean = project(points_3d, K.squeeze(0), R.squeeze(0), t.squeeze(0))

    def perturbed_points_2d(k1: float = 0.0):
        # Project with the TRUE cameras; the perturbation is only in the
        # calibration used for triangulation (or the lens distortion applied to
        # the observations).
        xy = project(points_3d, K.squeeze(0), R.squeeze(0), t.squeeze(0))
        if k1 != 0.0:
            xy = apply_radial_distortion(xy, K, k1)
        # add i.i.d. pixel noise
        xy = xy + torch.randn_like(xy) * noise_px
        return xy

    conditions = {
        "clean": {},
        "rot_0.5_deg": {"rot_std": 0.5},
        "rot_1.0_deg": {"rot_std": 1.0},
        "trans_5mm": {"trans_std": 0.005},
        "trans_10mm": {"trans_std": 0.010},
        "focal_1pct": {"focal_std": 0.01},
        "focal_2pct": {"focal_std": 0.02},
        "pp_3px": {"pp_std": 3.0},
        "pp_5px": {"pp_std": 5.0},
        "distortion_k1_0.10": {"k1": 0.10},
        "distortion_k1_0.30": {"k1": 0.30},
    }

    results = {}
    for name, cfg in conditions.items():
        k1 = cfg.pop("k1", 0.0)
        focal_std = cfg.get("focal_std", 0.0)
        pp_std = cfg.get("pp_std", 0.0)
        rot_std = cfg.get("rot_std", 0.0)
        trans_std = cfg.get("trans_std", 0.0)
        Kp, _, _ = perturb_intrinsics_with_delta(K, focal_std=focal_std, pp_std=pp_std)
        Rp, tp = perturb_extrinsics(R, t, rot_std=rot_std, trans_std=trans_std)
        xy = perturbed_points_2d(k1=k1)
        mpjpe = triangulate_and_evaluate(xy, Kp, Rp, tp, points_3d)
        results[name] = {"mpjpe_mm": round(mpjpe, 2)}
        print(f"{name}: MPJPE = {mpjpe:.2f} mm")
    return results


def main():
    parser = argparse.ArgumentParser(description="CPU calibration robustness diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--noise_px", type=float, default=0.0, help="Gaussian pixel noise applied to all 2D observations")
    parser.add_argument("--out_json", type=str, default="outputs/calibration_robustness_cpu_diagnostic.json")
    args = parser.parse_args()

    results = run_diagnostic(seed=args.seed, n_views=args.n_views, noise_px=args.noise_px)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
