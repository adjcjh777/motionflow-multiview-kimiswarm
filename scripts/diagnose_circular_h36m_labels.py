#!/usr/bin/env python3
"""Diagnose whether H36M 3D labels are circular (DLT-triangulated from input 2D).

For each frame, the script triangulates the stored 2D keypoints with the stored
camera parameters using unweighted DLT and compares the result to the stored
3D labels via MPJPE.  If the labels were produced by DLT, the per-frame MPJPE
will be near zero.

Usage:
    python scripts/diagnose_circular_h36m_labels.py <npz_path> [--output <json_path>]

Example:
    python scripts/diagnose_circular_h36m_labels.py data/h36m_hf/s_01_act_02_multiview.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motionflow_mv.fusion.triangulation import triangulate_dlt  # noqa: E402


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build P = K @ [R | t] from intrinsics and extrinsics.

    Args:
        K: (3, 3) intrinsics.
        R: (3, 3) rotation.
        t: (3,) translation (world-to-camera).

    Returns:
        P: (3, 4) projection matrix.
    """
    P = np.empty((3, 4), dtype=np.float64)
    P[:, :3] = K @ R
    P[:, 3] = K @ t
    return P


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute MPJPE (mm) between pred and gt, both (J, 3)."""
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def diagnose_circular_labels(npz_path: Path) -> dict:
    """Run the circular-label diagnostic on a single H36M .npz file."""
    data = np.load(npz_path, allow_pickle=True)

    required_keys = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required_keys - set(data.files)
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {missing}")

    points_2d = data["points_2d"]  # (F, V, J, 2)
    # confidences = data["confidences"]  # not used for unweighted DLT
    joints_3d = data["joints_3d"]  # (F, J, 3)
    camera_K = data["camera_K"]  # (V, 3, 3)
    camera_R = data["camera_R"]  # (V, 3, 3)
    camera_t = data["camera_t"]  # (V, 3)

    n_frames, n_views, n_joints, _ = points_2d.shape

    proj_matrices = np.stack(
        [build_projection_matrix(camera_K[v], camera_R[v], camera_t[v]) for v in range(n_views)],
        axis=0,
    )  # (V, 3, 4)

    per_frame_mpjpe = np.empty(n_frames, dtype=np.float64)
    for f in range(n_frames):
        j3d_dlt = np.empty((n_joints, 3), dtype=np.float64)
        for j in range(n_joints):
            pts = points_2d[f, :, j, :]  # (V, 2)
            # Skip views with NaN 2D coordinates (should not happen for H36M, but be safe).
            valid = ~np.isnan(pts).any(axis=-1)
            if valid.sum() < 2:
                # Cannot triangulate with fewer than 2 views; mark as NaN.
                j3d_dlt[j] = np.nan
                continue
            j3d_dlt[j] = triangulate_dlt(pts[valid], proj_matrices[valid])

        if np.isnan(j3d_dlt).any():
            per_frame_mpjpe[f] = np.nan
        else:
            per_frame_mpjpe[f] = compute_mpjpe(j3d_dlt, joints_3d[f])

    valid_mpjpe = per_frame_mpjpe[~np.isnan(per_frame_mpjpe)]
    result = {
        "npz_path": str(npz_path),
        "num_frames": int(n_frames),
        "num_views": int(n_views),
        "num_joints": int(n_joints),
        "mean_mpjpe_mm": float(np.mean(valid_mpjpe)) if len(valid_mpjpe) else float("nan"),
        "max_mpjpe_mm": float(np.max(valid_mpjpe)) if len(valid_mpjpe) else float("nan"),
        "min_mpjpe_mm": float(np.min(valid_mpjpe)) if len(valid_mpjpe) else float("nan"),
        "median_mpjpe_mm": float(np.median(valid_mpjpe)) if len(valid_mpjpe) else float("nan"),
        "num_nan_frames": int(np.isnan(per_frame_mpjpe).sum()),
        "per_frame_mpjpe_mm": per_frame_mpjpe.tolist(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether H36M 3D labels are DLT-triangulated from input 2D."
    )
    parser.add_argument("npz_path", type=Path, help="Path to H36M .npz file.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of per-frame MPJPE values to print at the head/tail (default: 10).",
    )
    args = parser.parse_args()

    if not args.npz_path.exists():
        raise FileNotFoundError(f"File not found: {args.npz_path}")

    result = diagnose_circular_labels(args.npz_path)

    print(f"File: {result['npz_path']}")
    print(f"Frames: {result['num_frames']}  Views: {result['num_views']}  Joints: {result['num_joints']}")
    print(f"Mean MPJPE: {result['mean_mpjpe_mm']:.6f} mm")
    print(f"Median MPJPE: {result['median_mpjpe_mm']:.6f} mm")
    print(f"Max MPJPE: {result['max_mpjpe_mm']:.6f} mm")
    print(f"Min MPJPE: {result['min_mpjpe_mm']:.6f} mm")
    if result["num_nan_frames"]:
        print(f"NaN frames: {result['num_nan_frames']}")

    per_frame = result["per_frame_mpjpe_mm"]
    print(f"\nFirst {args.head} per-frame MPJPEs (mm):")
    for i, val in enumerate(per_frame[: args.head]):
        print(f"  frame {i:04d}: {val:.6f} mm")
    if len(per_frame) > args.head:
        print(f"  ... ({len(per_frame) - args.head} frames omitted)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
