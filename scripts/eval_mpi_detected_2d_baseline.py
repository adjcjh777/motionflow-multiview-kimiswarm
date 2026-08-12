"""MPI-INF-3DHP detected-2D DLT baseline.

Triangulates the stored 2D keypoints (from an off-the-shelf 2D detector such as
RTMPose) using the calibrated cameras and compares the result to the true 3D
mocap ground truth.  Reports MPJPE and PA-MPJPE in millimetres.

Usage
-----
    python scripts/eval_mpi_detected_2d_baseline.py
    python scripts/eval_mpi_detected_2d_baseline.py --config configs/splits/mpi_inf_3dhp_detected_2d_baseline_smoke.yaml
    python scripts/eval_mpi_detected_2d_baseline.py --device cuda --unweighted

Output
------
    outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq


def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def evaluate_file(path: Path, device: str, weighted: bool, chunk_size: int = 8192) -> dict:
    """Run DLT on a single canonical .npz and return metrics (in mm)."""
    data = np.load(path)

    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    if points_2d.ndim != 4:
        raise ValueError(f"Expected points_2d shape (T,V,J,2), got {points_2d.shape} in {path}")

    T, V, J, _ = points_2d.shape
    P = build_projection_matrices(K, R, t)
    P_t = torch.from_numpy(P).to(device=device, dtype=torch.float64)

    pred_chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        p2d_chunk = torch.from_numpy(points_2d[start:end]).to(device=device, dtype=torch.float64)
        if weighted:
            w_chunk = torch.from_numpy(confidences[start:end]).to(device=device, dtype=torch.float64)
        else:
            w_chunk = torch.ones((end - start, V, J), device=device, dtype=torch.float64)
        X_t = triangulate_dlt_batched_lstsq(p2d_chunk, P_t, weights=w_chunk)
        pred_chunks.append(X_t.detach().cpu().numpy())

    X = np.concatenate(pred_chunks, axis=0)

    # Canonical _m.npz files store coordinates in metres; report in mm.
    to_mm = 1000.0
    report = compute_all_metrics(X * to_mm, joints_3d * to_mm)

    return {
        "dataset": path.stem,
        "path": str(path),
        "shape": {"T": X.shape[0], "J": X.shape[1], "V": P.shape[0]},
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
    }


def _load_paths_from_config(split: dict, key: str) -> list[str]:
    """Load a list of paths from either ``*_paths`` or ``*`` style YAML keys."""
    paths_key = f"{key}_paths"
    if paths_key in split:
        return list(split[paths_key])
    if key in split:
        return list(split[key])
    return []


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    if not values:
        return 0.0
    total = sum(v * w for v, w in zip(values, weights))
    return total / sum(weights)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPI-INF-3DHP detected-2D DLT baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/splits/mpi_inf_3dhp_detected_2d_baseline.yaml",
        help="YAML split file defining train/val/test paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mpi_rtmpose_detected_2d/dlt_baseline_detected_2d.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (cpu or cuda).",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Also compute unweighted DLT (slower; default: confidence-weighted only).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / args.config) as fh:
        split = yaml.safe_load(fh)

    train_paths = [repo_root / p for p in _load_paths_from_config(split, "train")]
    val_paths = [repo_root / p for p in _load_paths_from_config(split, "val")]
    test_paths = [repo_root / p for p in _load_paths_from_config(split, "test")]

    all_items: list[tuple[str, Path]] = []
    for label, paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths),
    ]:
        for p in paths:
            all_items.append((label, p))

    if not all_items:
        print("No paths found in config.")
        return

    print(f"MPI-INF-3DHP detected-2D DLT baseline ({len(train_paths)} train, "
          f"{len(val_paths)} val, {len(test_paths)} test files)")
    print(f"Device: {args.device}\n")

    per_file_weighted: list[dict] = []
    per_file_unweighted: list[dict] = []

    for split_label, path in all_items:
        if not path.exists():
            print(f"[{split_label}] {path.name}: MISSING, skipping")
            continue

        print(f"[{split_label}] {path.name}", end=" ", flush=True)
        res_weighted = evaluate_file(path, args.device, weighted=True)
        out_line = f"conf-DLT={res_weighted['mpjpe_mm']:.3f}mm"
        if args.unweighted:
            res_unweighted = evaluate_file(path, args.device, weighted=False)
            out_line += f"  unw-DLT={res_unweighted['mpjpe_mm']:.3f}mm"
        else:
            res_unweighted = None
        print(out_line)
        per_file_weighted.append({"split": split_label, **res_weighted})
        if res_unweighted is not None:
            per_file_unweighted.append({"split": split_label, **res_unweighted})

    if not per_file_weighted:
        print("No successful evaluations.")
        return

    def _summarise(entries: list[dict]) -> dict:
        simple = float(np.mean([e["mpjpe_mm"] for e in entries])) if entries else 0.0
        weights = [max(e["shape"]["T"], 1) for e in entries]
        weighted = _weighted_mean([e["mpjpe_mm"] for e in entries], weights) if entries else 0.0

        split_means = {}
        for split_name in ["train", "val", "test"]:
            split_entries = [e for e in entries if e["split"] == split_name]
            if split_entries:
                split_weights = [max(e["shape"]["T"], 1) for e in split_entries]
                split_means[split_name] = {
                    "simple_mean_mm": float(np.mean([e["mpjpe_mm"] for e in split_entries])),
                    "weighted_mean_mm": _weighted_mean(
                        [e["mpjpe_mm"] for e in split_entries], split_weights
                    ),
                    "per_file_mm": {e["dataset"]: e["mpjpe_mm"] for e in split_entries},
                }
        return {
            "simple_mean_mm": simple,
            "weighted_mean_mm": weighted,
            "per_split": split_means,
        }

    summary_weighted = _summarise(per_file_weighted)
    summary_unweighted = _summarise(per_file_unweighted) if per_file_unweighted else None

    payload: dict = {
        "unit": "mm",
        "protocol": "MPI-INF-3DHP detected-2D DLT baseline",
        "confidence_weighted": {
            "mean_mpjpe_mm": summary_weighted["simple_mean_mm"],
            "weighted_mean_mpjpe_mm": summary_weighted["weighted_mean_mm"],
            "per_file": per_file_weighted,
            "splits": summary_weighted["per_split"],
        },
    }
    if summary_unweighted is not None:
        payload["unweighted"] = {
            "mean_mpjpe_mm": summary_unweighted["simple_mean_mm"],
            "weighted_mean_mpjpe_mm": summary_unweighted["weighted_mean_mm"],
            "per_file": per_file_unweighted,
            "splits": summary_unweighted["per_split"],
        }

    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 70)
    print("MPI-INF-3DHP detected-2D DLT baseline summary")
    print("=" * 70)
    print(f"Conf-weighted DLT mean MPJPE: {summary_weighted['simple_mean_mm']:.3f} mm")
    if summary_unweighted is not None:
        print(f"Unweighted DLT mean MPJPE:    {summary_unweighted['simple_mean_mm']:.3f} mm")
    for split_name, split_summary in summary_weighted["per_split"].items():
        print(f"  {split_name}: {split_summary['simple_mean_mm']:.3f} mm "
              f"(weighted {split_summary['weighted_mean_mm']:.3f} mm)")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
