#!/usr/bin/env python
"""Run the full ICRA/CVPR benchmark protocol for an OmniMultiViewFusionV5 checkpoint.

Runs H36M and MPI-INF-3DHP evaluations (clean + robustness + variable views),
collects the JSON outputs, and writes a single summary JSON/CSV.

Usage
-----
    python scripts/run_full_v5_benchmark.py \
        --checkpoint outputs/omniview_fusion_v5_webbridge_multi.pth \
        --h36m data/webbridge/h36m_meters/s_09_acts_02_multiview_m.npz \
        --mpi data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --out outputs/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def run_eval(script: str, args: list[str]) -> Dict[str, Any]:
    """Run an eval script and load its JSON output."""
    cmd = [sys.executable, script, *args]
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")

    # Find --out_json argument.
    out_json = None
    for i, arg in enumerate(args):
        if arg == "--out_json" and i + 1 < len(args):
            out_json = args[i + 1]
            break
    if out_json is None:
        raise ValueError("Could not determine --out_json from eval args")

    with open(out_json, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full v5 benchmark protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--h36m", type=str, required=True, help="H36M .npz dataset")
    parser.add_argument("--mpi", type=str, required=True, help="MPI-INF-3DHP .npz dataset")
    parser.add_argument("--out", type=str, default="outputs/benchmark_results.json", help="Output summary JSON")
    parser.add_argument("--csv", type=str, default="outputs/benchmark_results.csv", help="Output summary CSV")
    args = parser.parse_args()

    h36m_json = "outputs/benchmark_h36m_tmp.json"
    h36m_csv = "outputs/benchmark_h36m_tmp.csv"
    mpi_json = "outputs/benchmark_mpi_tmp.json"
    mpi_csv = "outputs/benchmark_mpi_tmp.csv"

    results: Dict[str, Any] = {"checkpoint": args.checkpoint}

    print("Running H36M evaluation...")
    results["h36m"] = run_eval(
        "experiments/eval_omniview_fusion_v5_h36m.py",
        [
            "--checkpoint", args.checkpoint,
            "--dataset", args.h36m,
            "--run_robustness",
            "--run_variable_views",
            "--out_json", h36m_json,
            "--out_csv", h36m_csv,
        ],
    )

    print("Running MPI-INF-3DHP evaluation...")
    results["mpiinf3dhp"] = run_eval(
        "experiments/eval_omniview_fusion_v5_mpiinf3dhp.py",
        [
            "--checkpoint", args.checkpoint,
            "--dataset", args.mpi,
            "--run_robustness",
            "--run_variable_views",
            "--out_json", mpi_json,
            "--out_csv", mpi_csv,
        ],
    )

    # Summary metrics.
    summary = {
        "h36m_mpjpe_mm": results["h36m"].get("clean", {}).get("mpjpe"),
        "h36m_pa_mpjpe_mm": results["h36m"].get("clean", {}).get("pa_mpjpe"),
        "mpi_mpjpe_mm": results["mpiinf3dhp"].get("clean", {}).get("mpjpe"),
        "mpi_pa_mpjpe_mm": results["mpiinf3dhp"].get("clean", {}).get("pa_mpjpe"),
    }
    results["summary"] = summary

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = Path(args.csv)
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, f"{v:.2f}" if v is not None else "nan"])

    print(f"Summary -> {out_path}")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f} mm" if v is not None else f"  {k}: nan")


if __name__ == "__main__":
    main()
