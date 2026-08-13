"""Confidence-weighted DLT baseline for the AIST++ val file used in the
H36M+AIST mixed smoke split.

Reports MPJPE for the single AIST val clip and saves it to a JSON file so the
smoke result can be compared against a pure triangulation baseline.
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
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def evaluate_file(path: Path, device: str = "cpu", chunk_size: int = 8192) -> dict:
    data = np.load(path)

    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    T, V, J, _ = points_2d.shape
    P = build_projection_matrices(K, R, t)
    P_t = torch.from_numpy(P).to(device=device, dtype=torch.float64)

    pred_chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        p2d_chunk = torch.from_numpy(points_2d[start:end]).to(device=device, dtype=torch.float64)
        w_chunk = torch.from_numpy(confidences[start:end]).to(device=device, dtype=torch.float64)
        X_t = triangulate_dlt_batched_lstsq(p2d_chunk, P_t, weights=w_chunk)
        pred_chunks.append(X_t.detach().cpu().numpy())

    X = np.concatenate(pred_chunks, axis=0)
    to_mm = 1000.0
    report = compute_all_metrics(X * to_mm, joints_3d * to_mm)

    return {
        "dataset": path.stem,
        "path": str(path),
        "shape": {"T": int(T), "J": int(J), "V": int(V)},
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/splits/h36m_true_gt_v2_aist_mixed_smoke.yaml",
        help="YAML split file containing the AIST val path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/aist_mixed_smoke_dlt_baseline.json",
        help="Where to write the per-file JSON result.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / args.config) as fh:
        split = yaml.safe_load(fh)

    val_paths = split.get("val_paths", [])
    val_names = split.get("val_names", [])
    aist_val_paths = []
    for name, p in zip(val_names, val_paths):
        if "aist" in name.lower() or "aistpp" in p.lower():
            aist_val_paths.append(repo_root / p)

    if not aist_val_paths:
        raise RuntimeError("No AIST val path found in the provided split config.")

    results = []
    for path in aist_val_paths:
        print(f"Evaluating confidence-weighted DLT on {path.name} ...")
        res = evaluate_file(path, device=args.device)
        print(f"  MPJPE = {res['mpjpe_mm']:.3f} mm  PA-MPJPE = {res['pa_mpjpe_mm']:.3f} mm")
        results.append(res)

    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"unit": "mm", "per_file": results}, fh, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
