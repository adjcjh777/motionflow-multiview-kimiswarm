"""Audit a canonical multi-view .npz by reprojecting joints_3d into the cameras.

Projects the stored 3D labels through the stored cameras and compares the
result with the stored 2D keypoints.  This is the acceptance test for true 3D
ground truth:

* circular labels (``joints_3d == DLT(points_2d, cameras)``) reproject to the
  input 2D almost exactly (RMSE ~0 px);
* true mocap GT with the correct skeleton/units/frame alignment reprojects to
  the detected 2D within a few pixels (detector noise);
* a wrong joint mapping, unit, or frame alignment produces large errors
  (tens to hundreds of px).

Usage:
    python scripts/check_true_gt_reprojection.py data/h36m_hf/s_01_act_02_multiview.npz
    python scripts/check_true_gt_reprojection.py "data/h36m_true_npz/*.npz" --threshold 15
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def reproject(
    joints_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``(T, J, 3)`` world points into one camera.

    Returns ``(uv, depth)`` where ``uv`` is ``(T, J, 2)`` pixels and ``depth``
    is ``(T, J)`` camera-space z.
    """
    Xc = np.einsum("ij,tfj->tfi", R, joints_3d) + t[None, None, :]
    depth = Xc[..., 2]
    safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uv = np.stack(
        [fx * Xc[..., 0] / safe + cx, fy * Xc[..., 1] / safe + cy], axis=-1
    )
    return uv, depth


def check_npz(
    path: Path,
    conf_threshold: float = 0.5,
) -> Tuple[float, float, List[float]]:
    """Compute reprojection RMSE for one canonical npz.

    Returns ``(overall_rmse_px, inlier_fraction, per_view_rmse)`` where the
    errors are evaluated over joints with ``confidences > conf_threshold``
    (all joints if no confidence array is present).
    """
    data = np.load(path, allow_pickle=True)
    required = ("points_2d", "joints_3d", "camera_K", "camera_R", "camera_t")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    p2d = np.asarray(data["points_2d"], dtype=np.float64)
    j3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)
    conf = (
        np.asarray(data["confidences"], dtype=np.float64)
        if "confidences" in data.files
        else None
    )

    n_views = K.shape[0]
    if p2d.shape[1] != n_views:
        raise ValueError(
            f"{path}: points_2d has {p2d.shape[1]} views but cameras have {n_views}"
        )

    errs: List[float] = []
    per_view: List[float] = []
    n_inlier = 0
    n_total = 0
    for v in range(n_views):
        uv, depth = reproject(j3d, K[v], R[v], t[v])
        if np.any(depth <= 0):
            frac = float(np.mean(depth <= 0))
            print(f"  WARNING view {v}: {frac:.1%} of joints behind the camera")
        d = uv - p2d[:, v]
        e = np.linalg.norm(d, axis=-1)  # (T, J)
        if conf is not None:
            mask = conf[:, v] > conf_threshold
        else:
            mask = np.ones(e.shape, dtype=bool)
        if not mask.any():
            per_view.append(float("nan"))
            continue
        sel = e[mask]
        errs.append(sel)
        per_view.append(float(np.sqrt(np.mean(sel**2))))
        n_inlier += int(np.sum(sel < 15.0))
        n_total += int(sel.size)

    all_errs = np.concatenate(errs) if errs else np.zeros(0)
    overall = float(np.sqrt(np.mean(all_errs**2))) if all_errs.size else float("nan")
    frac = (n_inlier / n_total) if n_total else float("nan")
    return overall, frac, per_view


def main() -> Optional[int]:
    parser = argparse.ArgumentParser(
        description="Reprojection audit of canonical multi-view npz files."
    )
    parser.add_argument("paths", nargs="+", help="glob patterns or file paths to .npz")
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Max acceptable overall reprojection RMSE in px (default 15).",
    )
    args = parser.parse_args()

    paths: List[Path] = []
    for p in args.paths:
        if "*" in p:
            paths.extend(Path(f) for f in glob.glob(p, recursive=True))
        else:
            paths.append(Path(p))
    if not paths:
        print("No files matched.")
        return 1

    worst = 0.0
    for path in sorted(set(paths)):
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        overall, frac, per_view = check_npz(path)
        view_str = ", ".join(f"{e:.2f}" for e in per_view)
        print(f"{path}")
        print(f"  reprojection RMSE: {overall:.2f} px (views: [{view_str}])")
        print(f"  inlier fraction (<15 px): {frac:.3f}")
        worst = max(worst, overall)

    if worst > args.threshold:
        print(
            f"FAIL: worst RMSE {worst:.2f} px exceeds threshold "
            f"{args.threshold} px. Check joint mapping, units, and frame alignment."
        )
        return 1
    print(f"OK: worst RMSE {worst:.2f} px within threshold {args.threshold} px.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
