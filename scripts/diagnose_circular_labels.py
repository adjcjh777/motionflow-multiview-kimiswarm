#!/usr/bin/env python3
"""Diagnose whether H36M/WebBridge 3D labels are DLT triangulations of input 2D.

Loads one or more canonical .npz files (e.g. `data/h36m_hf/*_multiview.npz`),
re-triangulates the stored 2D keypoints with the stored cameras using the same
unweighted DLT routine, and reports MPJPE against the stored 3D labels.  If the
labels are a deterministic function of the inputs, MPJPE will be ~0 mm and the
correlation coefficient will be ~1.0.

Usage examples:
    python scripts/diagnose_circular_labels.py data/h36m_hf/s_01_act_02_multiview.npz
    python scripts/diagnose_circular_labels.py "data/h36m_hf/*.npz"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

# Allow importing project modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.triangulation import triangulate_dlt


def _build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices from intrinsics/extrinsics."""
    P = np.zeros((K.shape[0], 3, 4), dtype=np.float64)
    for v in range(K.shape[0]):
        Rt = np.concatenate([R[v], t[v][:, None]], axis=1)  # (3, 4)
        P[v] = K[v] @ Rt
    return P


def _mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root-centred MPJPE in mm (inputs in metres)."""
    pred = pred - pred.mean(axis=-2, keepdims=True)
    gt = gt - gt.mean(axis=-2, keepdims=True)
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def _direct_mje(pred: np.ndarray, gt: np.ndarray) -> float:
    """Direct per-joint L2 error in mm without root alignment."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def diagnose(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    p2d = data["points_2d"]
    j3d = data["joints_3d"]
    K = data["camera_K"]
    R = data["camera_R"]
    t = data["camera_t"]

    P = _build_projection_matrices(K, R, t)
    n_frames, n_views, n_joints, _ = p2d.shape

    re_tri = np.zeros_like(j3d)
    for f in range(n_frames):
        for j in range(n_joints):
            re_tri[f, j] = triangulate_dlt(p2d[f, :, j], P)

    direct_mm = _direct_mje(re_tri, j3d)
    root_mm = _mpjpe(re_tri, j3d)
    per_joint = np.linalg.norm(re_tri - j3d, axis=-1).mean(axis=0) * 1000.0

    print(f"{path}")
    print(f"  frames={n_frames}, views={n_views}, joints={n_joints}")
    print(f"  direct MJE (no root align): {direct_mm:.4f} mm")
    print(f"  root-aligned MPJPE:       {root_mm:.4f} mm")
    print(f"  max per-joint error:      {per_joint.max():.4f} mm")
    print(f"  median per-joint error:   {np.median(per_joint):.4f} mm")
    print(f"  mean per-joint error:     {per_joint.mean():.4f} mm")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for circular 3D labels in .npz files")
    parser.add_argument("paths", nargs="+", help="glob patterns or file paths to .npz files")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.paths:
        if "*" in p:
            paths.extend(Path(f) for f in glob.glob(p, recursive=True))
        else:
            paths.append(Path(p))

    if not paths:
        print("No files matched.")
        sys.exit(1)

    for path in sorted(set(paths)):
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        diagnose(path)


if __name__ == "__main__":
    main()
