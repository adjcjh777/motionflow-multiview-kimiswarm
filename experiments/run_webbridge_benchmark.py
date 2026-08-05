"""Run a unified multi-dataset benchmark over canonical WebBridge .npz files.

The harness reads a YAML manifest, calls ``experiments/eval_full_metrics.py`` for each
dataset, and writes a single CSV/JSON summary table.

Example
-------
    python experiments/run_webbridge_benchmark.py \
        --manifest configs/benchmark_webbridge_mpi_smoke.yaml \
        --out outputs/webbridge_benchmark_mpi_smoke
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run_eval(dataset_path: str, base_args: dict, output_json: Path) -> dict:
    """Call eval_full_metrics.py for one dataset and return the metrics dict."""
    cmd = [
        sys.executable,
        "experiments/eval_full_metrics.py",
        "--model", base_args["model"],
        "--dataset", dataset_path,
        "--checkpoint", base_args["checkpoint"],
        "--clip_len", str(base_args["clip_len"]),
        "--d", str(base_args["d"]),
        "--residual_hidden", str(base_args["residual_hidden"]),
        "--batch_size", str(base_args.get("batch_size", 8)),
        "--gt_scale", str(base_args.get("gt_scale", 1.0)),
        "--camera_scale", str(base_args.get("camera_scale", 1.0)),
        "--output_json", str(output_json),
    ]

    if base_args["model"] in {"crossview_residual", "crossview_residual_pp"}:
        cmd.extend(["--n_st_layers", str(base_args.get("n_st_layers", 2))])
    else:
        cmd.extend(["--n_temporal_layers", str(base_args.get("n_temporal_layers", 2))])

    # Optional extras for older model variants.
    for key in ("graph_layers", "k", "target_k", "min_views", "parents", "symmetry_pairs"):
        if key in base_args:
            cmd.extend([f"--{key}", str(base_args[key])])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"eval_full_metrics.py failed for {dataset_path}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    with open(output_json) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Unified WebBridge benchmark harness")
    parser.add_argument("--manifest", type=str, required=True, help="YAML manifest path")
    parser.add_argument("--out", type=str, required=True, help="Output prefix (no extension)")
    parser.add_argument("--device", type=str, default=None, help="CUDA device string (unused, kept for compatibility)")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Manifest YAML must contain a top-level dictionary")

    base = manifest.get("model_config", {})
    required = {"model", "checkpoint", "clip_len", "d", "residual_hidden"}
    missing = required - set(base.keys())
    if missing:
        raise ValueError(f"model_config missing required keys: {missing}")

    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets listed in manifest")

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in datasets:
        name = item["name"]
        path = item["path"]
        print(f"\n=== Evaluating {name}: {path} ===")
        output_json = out_prefix.parent / f"{out_prefix.name}_{name}.json"

        try:
            metrics = _run_eval(path, base, output_json)
        except RuntimeError as e:
            print(f"[ERROR] {name}: {e}")
            metrics = {}

        rows.append({
            "dataset": name,
            "path": path,
            "mpjpe_mm": metrics.get("mpjpe"),
            "pa_mpjpe_mm": metrics.get("pa_mpjpe"),
            "pck_50": metrics.get("pck@50mm"),
            "pck_100": metrics.get("pck@100mm"),
            "pck_150": metrics.get("pck@150mm"),
            "pck_auc": metrics.get("pck_auc"),
        })

    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump({
            "manifest": args.manifest,
            "model_config": base,
            "results": rows,
        }, f, indent=2)

    print(f"\n=== Summary written to {csv_path} and {json_path} ===")
    for row in rows:
        mpjpe = row["mpjpe_mm"]
        pa = row["pa_mpjpe_mm"]
        print(f"  {row['dataset']}: MPJPE={mpjpe:.2f} mm, PA-MPJPE={pa:.2f} mm" if mpjpe is not None else f"  {row['dataset']}: FAILED")


if __name__ == "__main__":
    main()
