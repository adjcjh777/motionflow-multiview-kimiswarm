"""Full AIST++ DLT baseline (confidence-weighted and unweighted).

Reads every canonical AIST++ multi-view .npz, triangulates 2D keypoints with
the stored calibrated cameras, and compares to the 3D ground truth. Reports
MPJPE and PA-MPJPE for both confidence-weighted and unweighted DLT.

Usage
-----
    python experiments/run_aistpp_full_dlt_baseline.py
    python experiments/run_aistpp_full_dlt_baseline.py --glob "data/webbridge/aistpp_canonical/*.npz" --device cpu
    python experiments/run_aistpp_full_dlt_baseline.py --output outputs/aistpp_full_dlt_baseline.json

Notes
-----
* AIST++ canonical files are in metres; reported values are converted to mm.
* Missing 2D observations (NaN) are masked out before triangulation.
* Aggregates are weighted by clip length in frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch as _torch

from experiments.baselines import build_projection_matrices, reprojection_error
from motionflow_mv.eval.metrics import pa_mpjpe


def dlt_from_views(points_2d: np.ndarray, confidences: np.ndarray,
                   P: np.ndarray, weighted: bool = True) -> np.ndarray:
    """Confidence-weighted DLT using batched torch SVD."""
    T, V, J, _ = points_2d.shape

    conf = _torch.from_numpy(confidences).permute(0, 2, 1)  # (T, J, V)
    zero_frames = conf.sum(dim=-1, keepdim=True) == 0
    conf = conf.clone()
    conf[zero_frames.expand_as(conf)] = 1.0

    p2d = _torch.from_numpy(points_2d).permute(0, 2, 1, 3)  # (T, J, V, 2)
    P_t = _torch.from_numpy(P)  # (V, 3, 4)

    nan_mask = _torch.isnan(p2d).any(dim=-1)  # (T, J, V)
    p2d = _torch.where(_torch.isnan(p2d), _torch.zeros_like(p2d), p2d)
    conf = conf.clone()
    conf[nan_mask] = 0.0

    if not weighted:
        conf = _torch.ones_like(conf)

    x = p2d[..., 0]
    y = p2d[..., 1]
    P0 = P_t[:, 0, :].unsqueeze(0).unsqueeze(0)
    P1 = P_t[:, 1, :].unsqueeze(0).unsqueeze(0)
    P2 = P_t[:, 2, :].unsqueeze(0).unsqueeze(0)
    row_x = x.unsqueeze(-1) * P2 - P0
    row_y = y.unsqueeze(-1) * P2 - P1

    A = _torch.stack([row_x, row_y], dim=-2)
    w_sqrt = conf.unsqueeze(-1).unsqueeze(-1).sqrt()
    A = A * w_sqrt
    A = A.reshape(T, J, 2 * V, 4)

    _, _, Vh = _torch.linalg.svd(A)
    X = Vh[..., -1, :]
    pred = X[..., :3] / X[..., 3:4]
    return pred.numpy()


def evaluate_npz(path: Path, weighted: bool = True) -> dict:
    data = np.load(path)
    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    P = build_projection_matrices(K, R, t)
    pred = dlt_from_views(points_2d, confidences, P, weighted=weighted)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    joints_3d = np.where(np.isfinite(joints_3d), joints_3d, 0.0)

    e = np.linalg.norm(pred - joints_3d, axis=-1)
    reproj = reprojection_error(pred, points_2d, P, confidences)
    return {
        "shape": {"T": points_2d.shape[0], "V": points_2d.shape[1], "J": points_2d.shape[2]},
        "metrics": {
            "mpjpe": float(e.mean()),
            "pa_mpjpe": pa_mpjpe(pred, joints_3d),
            "reproj_px": float(np.nanmean(reproj)),
        },
    }


def weighted_aggregate(records: list[dict]) -> dict:
    total_frames = sum(r["shape"]["T"] for r in records)
    if total_frames == 0:
        return {}

    def weighted(key: str) -> float:
        return sum(r["metrics"][key] * r["shape"]["T"] for r in records) / total_frames

    return {
        "mpjpe_m": weighted("mpjpe"),
        "pa_mpjpe_m": weighted("pa_mpjpe"),
        "reproj_px": weighted("reproj_px"),
        "total_frames": total_frames,
        "n_clips": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full AIST++ DLT baseline.")
    parser.add_argument(
        "--glob",
        type=str,
        default="data/webbridge/aistpp_canonical/*.npz",
        help="Glob pattern for .npz files to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/aistpp_full_dlt_baseline.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (cpu or cuda). Default: cpu.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    files = sorted(repo_root.glob(args.glob))
    if not files:
        print(f"No .npz files found for pattern: {args.glob}")
        return

    print(f"Found {len(files)} AIST++ .npz files. Running DLT baselines...")

    weighted_records = []
    unweighted_records = []
    missing = []

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", end=" ", flush=True)
        try:
            rec_w = evaluate_npz(f, weighted=True)
            rec_w["path"] = str(f)
            rec_w["clip"] = f.name
            weighted_records.append(rec_w)

            rec_u = evaluate_npz(f, weighted=False)
            rec_u["path"] = str(f)
            rec_u["clip"] = f.name
            unweighted_records.append(rec_u)

            print(f"MPJPE(w)={rec_w['metrics']['mpjpe']*1000:.2f} mm  MPJPE(u)={rec_u['metrics']['mpjpe']*1000:.2f} mm")
        except Exception as exc:
            print(f"FAILED: {exc}")
            missing.append(str(f))

    overall_w = weighted_aggregate(weighted_records)
    overall_u = weighted_aggregate(unweighted_records)

    print("\n" + "=" * 70)
    print("AIST++ full DLT baseline summary")
    print("=" * 70)
    print(f"{'Metric':<25} {'Confidence-weighted':>20} {'Unweighted':>15}")
    print("-" * 70)
    print(f"{'Clips':<25} {overall_w['n_clips']:>20} {overall_u['n_clips']:>15}")
    print(f"{'Total frames':<25} {overall_w['total_frames']:>20} {overall_u['total_frames']:>15}")
    print(f"{'MPJPE (mm)':<25} {overall_w['mpjpe_m']*1000:>20.2f} {overall_u['mpjpe_m']*1000:>15.2f}")
    print(f"{'PA-MPJPE (mm)':<25} {overall_w['pa_mpjpe_m']*1000:>20.2f} {overall_u['pa_mpjpe_m']*1000:>15.2f}")
    print(f"{'Reprojection error (px)':<25} {overall_w['reproj_px']:>20.2f} {overall_u['reproj_px']:>15.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_clips": len(files),
        "weighted": overall_w,
        "unweighted": overall_u,
        "per_clip_weighted": weighted_records,
        "per_clip_unweighted": unweighted_records,
        "failed": missing,
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
