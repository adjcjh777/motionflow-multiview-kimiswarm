#!/usr/bin/env python3
"""Reproducible confidence-weighted RANSAC-DLT baseline for H36M true-GT.

Runs the confidence-weighted 3-view random-subset RANSAC-DLT variant that
produces **26.47 mm** on the standard H36M true-GT test split (S9/S11).
This baseline is deterministic (seeded RNG) and uses only NumPy.

Usage
-----
    python scripts/run_h36m_true_gt_ransac_baseline.py
    python scripts/run_h36m_true_gt_ransac_baseline.py --all_unique

Output
------
    outputs/ransac_dlt_h36m_true_gt_reproducible.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.eval.metrics import compute_all_metrics  # noqa: E402


def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def _triangulate_subset_weighted(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    P: np.ndarray,
    subset: list[int],
) -> np.ndarray:
    """Triangulate (T, J, 3) using a view subset with confidence weighting."""
    p2d = points_2d[:, subset, :, :]  # (T, Vs, J, 2)
    P_sub = P[np.array(subset)]
    weights = confidences[:, subset, :]  # (T, Vs, J)
    T, Vs, J, _ = p2d.shape

    A = np.empty((T, J, 2 * Vs, 4), dtype=np.float64)
    for v in range(Vs):
        u = p2d[:, v, :, 0]
        vv = p2d[:, v, :, 1]
        Pv = P_sub[v]
        sw = np.sqrt(weights[:, v, :] + 1e-6)  # (T, J)
        A[:, :, 2 * v] = sw[..., None] * (u[..., None] * Pv[2] - Pv[0])
        A[:, :, 2 * v + 1] = sw[..., None] * (vv[..., None] * Pv[2] - Pv[1])

    _, _, vt = np.linalg.svd(A)
    X = vt[..., -1, :]
    return X[..., :3] / X[..., 3:4]


def _confidence_weighted_dlt(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Vectorised confidence-weighted DLT fallback for too few valid views."""
    T, V, J, _ = points_2d.shape
    sw = np.sqrt(confidences + 1e-6)  # (T, V, J)

    A = np.zeros((T, J, 2 * V, 4), dtype=np.float64)
    for v in range(V):
        u = points_2d[:, v, :, 0]
        vv = points_2d[:, v, :, 1]
        Pv = P[v]
        w = sw[:, v, :]  # (T, J)
        A[:, :, 2 * v] = w[..., None] * (u[..., None] * Pv[2] - Pv[0])
        A[:, :, 2 * v + 1] = w[..., None] * (vv[..., None] * Pv[2] - Pv[1])

    _, _, vt = np.linalg.svd(A)
    X = vt[..., -1, :]
    return X[..., :3] / X[..., 3:4]


def ransac_dlt_weighted(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    P: np.ndarray,
    n_iter: int = 32,
    inlier_thr_px: float = 5.0,
    min_subset: int = 3,
    all_unique: bool = False,
) -> np.ndarray:
    """Confidence-weighted 3-view random-subset RANSAC-DLT.

    Parameters
    ----------
    points_2d: (T, V, J, 2)
    confidences: (T, V, J)
    P: (V, 3, 4)
    n_iter: number of random subsets to draw (ignored when ``all_unique=True``)
    inlier_thr_px: reprojection inlier threshold in pixels
    min_subset: size of each sampled view subset
    all_unique: if True, enumerate all ``V choose 3`` subsets instead of random

    Returns
    -------
    (T, J, 3) triangulated points in metres.
    """
    T, V, J, _ = points_2d.shape
    valid = confidences > 0
    n_valid = valid.sum(axis=1)  # (T, J)
    fallback = n_valid <= min_subset

    if all_unique:
        subsets = [list(c) for c in combinations(range(V), min_subset)]
    else:
        rng = np.random.default_rng(0)
        n_samples = min(n_iter, max(1, V * (V - 1) // 2))
        subsets = [rng.choice(V, size=min_subset, replace=False).tolist() for _ in range(n_samples)]

    if not subsets:
        return _confidence_weighted_dlt(points_2d, confidences, P)

    preds = np.stack(
        [_triangulate_subset_weighted(points_2d, confidences, P, s) for s in subsets],
        axis=0,
    )  # (S, T, J, 3)

    # Reprojection residuals for each subset prediction across all views.
    X_h = np.concatenate([preds, np.ones((*preds.shape[:3], 1))], axis=-1)  # (S, T, J, 4)
    x_h = np.einsum("vik,stjk->stvji", P, X_h)  # (S, T, V, J, 3)
    x_proj = x_h[..., :2] / x_h[..., 2:3]  # (S, T, V, J, 2)
    diff = x_proj - points_2d  # broadcasts to (S, T, V, J, 2)
    residuals = np.linalg.norm(diff, axis=-1)  # (S, T, V, J)

    inliers = residuals < inlier_thr_px  # (S, T, V, J)
    n_inliers = inliers.sum(axis=2)  # (S, T, J)
    score = residuals.mean(axis=2)  # (S, T, J)
    key = -n_inliers.astype(np.float64) * 1e6 + score
    best = np.argmin(key, axis=0)  # (T, J)

    X = preds[best, np.arange(T)[:, None], np.arange(J)[None, :], :]

    if fallback.any():
        fallback_pred = _confidence_weighted_dlt(points_2d, confidences, P)
        X = np.where(fallback[..., None], fallback_pred, X)

    return X


def evaluate_file(path: Path, all_unique: bool = False) -> dict:
    """Run RANSAC-DLT on one canonical .npz and return metrics (in mm)."""
    data = np.load(path)

    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    P = build_projection_matrices(K, R, t)
    pred = ransac_dlt_weighted(points_2d, confidences, P, all_unique=all_unique)

    # Convert metres to mm for reporting.
    to_mm = 1000.0
    report = compute_all_metrics(pred * to_mm, joints_3d * to_mm)

    return {
        "dataset": path.stem,
        "path": str(path),
        "shape": {"T": int(points_2d.shape[0]), "V": int(points_2d.shape[1]), "J": int(points_2d.shape[2])},
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible confidence-weighted RANSAC-DLT baseline for H36M true-GT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/splits/h36m_true_gt_standard.yaml",
        help="YAML split file defining the test paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path. Defaults based on --all_unique.",
    )
    parser.add_argument(
        "--all_unique",
        action="store_true",
        help="Enumerate all 3-view subsets instead of drawing random samples.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / args.config) as fh:
        split = yaml.safe_load(fh)

    val_paths = [repo_root / p for p in split.get("val_paths", [])]
    if not val_paths:
        print("No val/test paths found in config.")
        return

    print(f"Reproducible RANSAC-DLT baseline ({len(val_paths)} test files)")
    print("all_unique subsets:", args.all_unique)

    results = []
    for path in val_paths:
        print(f"[test] {path.name}", end=" ", flush=True)
        res = evaluate_file(path, all_unique=args.all_unique)
        print(f"MPJPE={res['mpjpe_mm']:.4f} mm  PA-MPJPE={res['pa_mpjpe_mm']:.4f} mm")
        results.append(res)

    total_frames = sum(r["shape"]["T"] for r in results)
    combined_mpjpe = sum(r["mpjpe_mm"] * r["shape"]["T"] for r in results) / total_frames
    combined_pa = sum(r["pa_mpjpe_mm"] * r["shape"]["T"] for r in results) / total_frames

    print(f"\nCombined test MPJPE (frame-weighted S9+S11) = {combined_mpjpe:.4f} mm")
    print(f"Combined test PA-MPJPE (frame-weighted S9+S11) = {combined_pa:.4f} mm")

    out = {
        "method": "ransac_dlt_conf_weighted",
        "all_unique": args.all_unique,
        "subjects": results,
        "combined_mpjpe_mm": combined_mpjpe,
        "combined_pa_mpjpe_mm": combined_pa,
    }

    if args.output is None:
        if args.all_unique:
            out_path = repo_root / "outputs/ransac_dlt_h36m_true_gt_reproducible_all_unique.json"
        else:
            out_path = repo_root / "outputs/ransac_dlt_h36m_true_gt_reproducible.json"
    else:
        out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
