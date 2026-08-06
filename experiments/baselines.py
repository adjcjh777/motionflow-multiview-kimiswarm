"""Benchmark classic triangulation baselines on prepared multi-view .npz datasets.

Implemented baselines
---------------------
* ``dlt``               - confidence-weighted Direct Linear Transform.
* ``ransac_dlt``        - minimal-view-set sampling with inlier scoring.
* ``robust``            - iterative re-weighted least squares (Huber) DLT.
* ``temporal``          - DLT followed by a simple temporal moving-average.

Usage
-----
    # discover all `data/**/*.npz` files and run every baseline
    python experiments/baselines.py

    # run on specific files, first 1000 frames only
    python experiments/baselines.py --datasets data/h36m_hf/*.npz --max_frames 1000

    # write a JSON table of results
    python experiments/baselines.py --output outputs/baseline_results.json

Important findings (verified 2026-08-04, first 200 frames of each .npz)
----------------------------------------------------------------------
* H36M subject 1 is internally consistent: DLT re-creates the per-frame 3D
  ground truth with ~1.9 mm MPJPE, matching the DLT baseline in docs/design_v3.md.
* RANSAC-DLT falls back to DLT when there are only 4 views; on this clean data
  it is identical to DLT.
* Robust Huber weighting stays close to DLT (~3.6 mm) on clean H36M.
* Temporal smoothing trades per-frame accuracy for smoothness and is expected to
  lag the per-frame GT.
* Shelf_Seq1/pseudogt.npz and H36M s_09 show large 3D errors because the stored
  3D ground truth and the bundled cameras are not on the same scale/coordinate
  frame.  The script still runs, but those datasets are not yet suitable for
  benchmarking geometric accuracy without additional alignment.

Environment notes
-----------------
* The script is self-contained and depends only on NumPy.
* On the Windows conda environment used in this repo, NumPy's BLAS DLLs live
  under ``<env>/Library/bin``; add that directory to ``PATH`` if SVD fails.
* ``KMP_DUPLICATE_LIB_OK=TRUE`` is also needed to avoid the OpenMP runtime
  conflict in this environment.
* 3D errors are reported in each dataset's native length unit (mm for the
  prepared H36M files, cm for the Shelf pseudogt).  Compare methods on the same
  dataset rather than across datasets.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Minimal NumPy-only helpers (keeps the script PyTorch-free)
# --------------------------------------------------------------------------- #

def triangulate_dlt(points_2d: np.ndarray, proj_matrices: np.ndarray,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """Triangulate one 3D point from N calibrated views.

    Args:
        points_2d: (N, 2) array of 2D keypoints.
        proj_matrices: (N, 3, 4) projection matrices P_i.
        weights: optional (N,) array for confidence weighting.

    Returns:
        (3,) array, the triangulated 3D point in world coordinates.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    proj_matrices = np.asarray(proj_matrices, dtype=np.float64)
    n_views = points_2d.shape[0]

    if weights is None:
        weights = np.ones(n_views, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(n_views)
        weights = np.sqrt(weights + 1e-6)

    A = []
    for (u, v), P, w in zip(points_2d, proj_matrices, weights):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    A = np.stack(A)  # (2*N, 4)

    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Per Joint Position Error."""
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def _align_rigid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return (Xc @ R) + Y.mean(axis=0)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-frame Procrustes-aligned MPJPE."""
    T = pred.shape[0]
    aligned = np.zeros_like(pred)
    for t in range(T):
        aligned[t] = _align_rigid(pred[t], gt[t])
    return mpjpe(aligned, gt)


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #

def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def reprojection_error(X: np.ndarray, points_2d: np.ndarray, P: np.ndarray,
                       confidences: np.ndarray | None = None) -> np.ndarray:
    """Per-joint reprojection error in pixels.

    Args:
        X: (T, J, 3) triangulated points.
        points_2d: (T, V, J, 2) observations.
        P: (V, 3, 4) projection matrices.
        confidences: optional (T, V, J) mask; zero values are ignored.

    Returns:
        (T, J) error array.
    """
    X_h = np.concatenate([X, np.ones((*X.shape[:2], 1))], axis=-1)  # (T, J, 4)
    x_h = np.einsum("vik,tjk->tvji", P, X_h)  # (T, V, J, 3)
    x_proj = x_h[..., :2] / x_h[..., 2:3]
    diff = x_proj - points_2d  # (T, V, J, 2)
    err = np.linalg.norm(diff, axis=-1)  # (T, V, J)
    if confidences is not None:
        mask = confidences > 0  # (T, V, J)
        err = np.where(mask, err, np.nan)
    return np.nanmean(err, axis=1)


# --------------------------------------------------------------------------- #
# Baseline triangulation methods
# --------------------------------------------------------------------------- #

def baseline_dlt(points_2d: np.ndarray, confidences: np.ndarray,
                 P: np.ndarray) -> np.ndarray:
    """Confidence-weighted DLT for each frame and joint.

    Args:
        points_2d: (T, V, J, 2)
        confidences: (T, V, J)
        P: (V, 3, 4)

    Returns:
        (T, J, 3) triangulated points.
    """
    T, V, J, _ = points_2d.shape
    X = np.zeros((T, J, 3), dtype=np.float64)
    for t in range(T):
        for j in range(J):
            w = confidences[t, :, j]
            if w.sum() == 0:
                w = np.ones_like(w)
            X[t, j] = triangulate_dlt(points_2d[t, :, j], P, weights=w)
    return X


def baseline_ransac_dlt(points_2d: np.ndarray, confidences: np.ndarray,
                        P: np.ndarray, n_iter: int = 32,
                        inlier_thresh_px: float = 5.0,
                        min_subset: int = 3) -> np.ndarray:
    """Sample minimal view subsets, triangulate, and pick the set with most inliers.

    Falls back to confidence-weighted DLT when there are too few views for
    meaningful sampling.
    """
    T, V, J, _ = points_2d.shape
    X = np.zeros((T, J, 3), dtype=np.float64)
    for t in range(T):
        for j in range(J):
            w = confidences[t, :, j]
            valid = w > 0
            n_valid = valid.sum()
            if n_valid <= min_subset or V <= 4:
                X[t, j] = triangulate_dlt(points_2d[t, :, j], P, weights=w)
                continue

            valid_idxs = np.where(valid)[0]
            best_inliers = -1
            best_score = float("inf")
            best_X = None
            rng = np.random.default_rng(0)
            n_samples = min(n_iter, max(1, len(valid_idxs) * (len(valid_idxs) - 1) // 2))
            for _ in range(n_samples):
                subset = rng.choice(valid_idxs, size=min_subset, replace=False)
                Xc = triangulate_dlt(points_2d[t, subset, j], P[subset])
                # score by inliers across *all* valid views
                X_h = np.append(Xc, 1.0)
                x_h = P @ X_h  # (V, 3)
                x_proj = x_h[:, :2] / x_h[:, 2:3]
                residuals = np.linalg.norm(x_proj - points_2d[t, :, j], axis=-1)
                inliers = (residuals < inlier_thresh_px) & valid
                n_inliers = inliers.sum()
                score = residuals[valid].mean()
                if n_inliers > best_inliers or (n_inliers == best_inliers and score < best_score):
                    best_inliers = n_inliers
                    best_score = score
                    best_X = Xc

            if best_X is None:
                best_X = triangulate_dlt(points_2d[t, :, j], P, weights=w)
            X[t, j] = best_X
    return X


def baseline_robust(points_2d: np.ndarray, confidences: np.ndarray,
                    P: np.ndarray, n_iter: int = 5,
                    huber_delta_px: float = 5.0) -> np.ndarray:
    """Iteratively re-weighted DLT using a Huber-like residual weighting."""
    T, V, J, _ = points_2d.shape
    X = np.zeros((T, J, 3), dtype=np.float64)
    for t in range(T):
        for j in range(J):
            w = confidences[t, :, j].copy().astype(np.float64)
            valid = w > 0
            if valid.sum() == 0:
                w = np.ones(V, dtype=np.float64)
                valid = w > 0
            X[t, j] = triangulate_dlt(points_2d[t, :, j], P, weights=w)
            for _ in range(n_iter):
                X_h = np.append(X[t, j], 1.0)
                x_h = P @ X_h
                x_proj = x_h[:, :2] / x_h[:, 2:3]
                residuals = np.linalg.norm(x_proj - points_2d[t, :, j], axis=-1)
                # Huber-ish weights
                weights = np.ones(V, dtype=np.float64)
                mask = residuals > huber_delta_px
                weights[mask] = huber_delta_px / residuals[mask]
                weights = weights * w
                if weights.sum() == 0:
                    weights = np.ones(V, dtype=np.float64)
                X[t, j] = triangulate_dlt(points_2d[t, :, j], P, weights=weights)
    return X


def baseline_temporal(points_2d: np.ndarray, confidences: np.ndarray,
                      P: np.ndarray, window: int = 5) -> np.ndarray:
    """DLT per frame followed by a simple temporal moving-average smoother."""
    X = baseline_dlt(points_2d, confidences, P)
    if window <= 1:
        return X
    # Simple centered moving average (mirror at boundaries).
    kernel = np.ones(window) / window
    X_smooth = np.copy(X)
    for j in range(X.shape[1]):
        for c in range(3):
            X_smooth[:, j, c] = np.convolve(X[:, j, c], kernel, mode="same")
    # Fix boundaries with smaller windows to avoid amplitude loss.
    half = window // 2
    for i in range(half):
        if i < X.shape[0]:
            X_smooth[i] = X[: i + half + 1].mean(axis=0)
            X_smooth[-(i + 1)] = X[-(i + half + 1) :].mean(axis=0)
    return X_smooth


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def evaluate_method(name: str, pred: np.ndarray, gt: np.ndarray,
                    points_2d: np.ndarray, P: np.ndarray,
                    confidences: np.ndarray) -> dict:
    """Return a dict of metrics for one baseline prediction."""
    e = np.linalg.norm(pred - gt, axis=-1)
    reproj = reprojection_error(pred, points_2d, P, confidences)
    return {
        "method": name,
        "mpjpe": float(np.mean(e)),
        "pa_mpjpe": pa_mpjpe(pred, gt),
        "reproj_px": float(np.nanmean(reproj)),
    }


def benchmark_dataset(path: Path, max_frames: int | None = None) -> dict:
    """Load one .npz dataset, run all baselines, and return metrics."""
    data = np.load(path)
    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    if points_2d.ndim == 4:
        T, V, J, _ = points_2d.shape
    else:
        raise ValueError(f"Unsupported points_2d shape {points_2d.shape} in {path}")

    if max_frames is not None and T > max_frames:
        points_2d = points_2d[:max_frames]
        confidences = confidences[:max_frames]
        joints_3d = joints_3d[:max_frames]
        T = max_frames

    P = build_projection_matrices(K, R, t)

    results = {
        "dataset": str(path),
        "shape": {"T": T, "V": V, "J": J},
        "methods": {},
    }

    dlt_pred = baseline_dlt(points_2d, confidences, P)
    results["methods"]["dlt"] = evaluate_method(
        "dlt", dlt_pred, joints_3d, points_2d, P, confidences
    )

    ransac_pred = baseline_ransac_dlt(points_2d, confidences, P)
    results["methods"]["ransac_dlt"] = evaluate_method(
        "ransac_dlt", ransac_pred, joints_3d, points_2d, P, confidences
    )

    robust_pred = baseline_robust(points_2d, confidences, P)
    results["methods"]["robust"] = evaluate_method(
        "robust", robust_pred, joints_3d, points_2d, P, confidences
    )

    temporal_pred = baseline_temporal(points_2d, confidences, P, window=5)
    results["methods"]["temporal"] = evaluate_method(
        "temporal", temporal_pred, joints_3d, points_2d, P, confidences
    )

    return results


def print_results(results: list[dict]) -> None:
    """Pretty-print a results table."""
    header = f"{'dataset':<40} {'method':<12} {'MPJPE':>10} {'PA-MPJPE':>10} {'Reproj(px)':>12}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        dataset = Path(r["dataset"]).name
        for method, metrics in r["methods"].items():
            print(
                f"{dataset:<40} {method:<12} "
                f"{metrics['mpjpe']:>10.4f} "
                f"{metrics['pa_mpjpe']:>10.4f} "
                f"{metrics['reproj_px']:>12.4f}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark classic triangulation baselines on .npz datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Paths or globs of prepared .npz datasets. Defaults to data/**/*.npz.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Limit each dataset to the first N frames (for quick verification).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file path to save the results table.",
    )
    args = parser.parse_args()

    if args.datasets is None:
        datasets = sorted(Path("data").glob("**/*.npz"))
    else:
        datasets = []
        for pat in args.datasets:
            p = Path(pat)
            if p.is_file():
                datasets.append(p)
            else:
                datasets.extend(Path(".").glob(pat))
        datasets = sorted(set(datasets))

    if not datasets:
        print("No .npz datasets found.")
        return

    print(f"Benchmarking {len(datasets)} dataset(s)...")
    results = []
    for ds in datasets:
        print(f"  {ds}")
        results.append(benchmark_dataset(ds, max_frames=args.max_frames))

    print_results(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
