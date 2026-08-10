"""Run a DLT triangulation baseline on MPI-INF-3DHP multiview .npz files.

This script triangulates the stored 2D keypoints using the stored calibrated
cameras and compares the result to the true 3D ground truth.  It reports
MPJPE and PA-MPJPE in both millimetres and metres so the numbers can be
compared directly with H36M-style benchmarks.

Usage
-----
    python scripts/run_mpi_dlt_baseline.py
    python scripts/run_mpi_dlt_baseline.py --glob "data/webbridge/mpi_inf_3dhp/*_v14_multiview_m.npz"
    python scripts/run_mpi_dlt_baseline.py --glob "data/webbridge/mpi_inf_3dhp/*.npz" --output outputs/mpi_dlt_baseline.json

Notes
-----
* MPI-INF-3DHP is stored in **metres** in the prepared .npz files.
* The script uses the existing batched DLT implementation in
  ``motionflow_mv/fusion/triangulation.py`` and the metrics in
  ``motionflow_mv/eval/metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq
from motionflow_mv.eval.metrics import compute_all_metrics


def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.concatenate([R[v], t[v].reshape(3, 1)], axis=1)
        P[v] = K[v] @ Rt
    return P


def triangulate_file(path: Path, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one .npz and return (predicted, ground_truth, metadata)."""
    data = np.load(path)

    points_2d = np.asarray(data["points_2d"], dtype=np.float64)
    confidences = np.asarray(data["confidences"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d"], dtype=np.float64)
    K = np.asarray(data["camera_K"], dtype=np.float64)
    R = np.asarray(data["camera_R"], dtype=np.float64)
    t = np.asarray(data["camera_t"], dtype=np.float64)

    if points_2d.ndim != 4:
        raise ValueError(f"Expected points_2d shape (T,V,J,2), got {points_2d.shape} in {path}")

    P = build_projection_matrices(K, R, t)

    # torch tensors on requested device
    points_2d_t = torch.from_numpy(points_2d).to(device=device, dtype=torch.float64)
    P_t = torch.from_numpy(P).to(device=device, dtype=torch.float64)
    conf_t = torch.from_numpy(confidences).to(device=device, dtype=torch.float64)

    # triangulate_dlt_batched_lstsq expects (N, V, J, 2) and (V, 3, 4) or (N, V, 3, 4)
    X_t = triangulate_dlt_batched_lstsq(points_2d_t, P_t, weights=conf_t)
    X = X_t.detach().cpu().numpy()

    meta = {"path": str(path), "shape": {"T": X.shape[0], "J": X.shape[1], "V": P.shape[0]}}
    return X, joints_3d, meta


def _is_metres(path: Path) -> bool:
    """Infer whether a canonical MPI .npz stores coordinates in metres.

    The canonical WebBridge MPI-INF-3DHP files with the ``_m`` suffix are in
    metres; the legacy non-``_m`` files are in millimetres.
    """
    return "_m.npz" in path.name or "_m_" in path.name


def evaluate_file(path: Path, device: str = "cpu") -> dict:
    """Run DLT and compute metrics for one .npz file."""
    pred, gt, meta = triangulate_file(path, device=device)

    report = compute_all_metrics(pred, gt)
    # The _m files are metres; legacy files are millimetres.  Normalise to mm.
    to_mm = 1000.0 if _is_metres(path) else 1.0
    result = {
        "dataset": Path(path).name,
        "path": meta["path"],
        "shape": meta["shape"],
        "unit": "m" if _is_metres(path) else "mm",
        "mpjpe_m": float(report["mpjpe"]),
        "pa_mpjpe_m": float(report["pa_mpjpe"]),
        "mpjpe_mm": float(report["mpjpe"]) * to_mm,
        "pa_mpjpe_mm": float(report["pa_mpjpe"]) * to_mm,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DLT triangulation baseline on MPI-INF-3DHP .npz files.")
    parser.add_argument("--glob", type=str, default="data/webbridge/mpi_inf_3dhp/*_m.npz",
                        help="Glob pattern for .npz files to evaluate. Default: canonical metre files.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file path to save the results table.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device to use (cpu or cuda).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    files = sorted(repo_root.glob(args.glob))

    if not files:
        print(f"No .npz files found for pattern: {args.glob}")
        return

    print(f"Evaluating DLT baseline on {len(files)} MPI-INF-3DHP file(s)...\n")
    results = []
    for f in files:
        print(f"  {f.name}", end=" ", flush=True)
        try:
            res = evaluate_file(f, device=args.device)
            results.append(res)
            print(f"MPJPE={res['mpjpe_mm']:.3f}mm  PA-MPJPE={res['pa_mpjpe_mm']:.3f}mm")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not results:
        print("No successful evaluations.")
        return

    mean_mpjpe = float(np.mean([r["mpjpe_mm"] for r in results]))
    mean_pa = float(np.mean([r["pa_mpjpe_mm"] for r in results]))

    print("\n" + "=" * 70)
    print("MPI-INF-3DHP DLT baseline summary (GT 2D -> triangulate -> 3D GT)")
    print("=" * 70)
    print(f"{'Dataset':<45} {'MPJPE (mm)':>12} {'PA-MPJPE (mm)':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['dataset']:<45} {r['mpjpe_mm']:>12.3f} {r['pa_mpjpe_mm']:>15.3f}")
    print("-" * 70)
    print(f"{'Mean':<45} {mean_mpjpe:>12.3f} {mean_pa:>15.3f}")
    print("\nAll numbers are reported in millimetres for easy comparison.")
    print("Canonical files (names ending in _m.npz) store coordinates in metres.")

    # Non-circularity sanity check.
    non_circular = all(r["mpjpe_mm"] > 0.1 for r in results)
    if non_circular:
        print("Circularity check: PASSED (MPJPE >> 0 for every file).")
    else:
        print("Circularity check: FAILED (at least one file has MPJPE ~= 0).")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "unit": "mm",
            "mean_mpjpe_mm": mean_mpjpe,
            "mean_pa_mpjpe_mm": mean_pa,
            "per_file": results,
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
