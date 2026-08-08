#!/usr/bin/env python3
"""YAML-manifest benchmark driver for OmniMultiViewFusionV5 checkpoints.

This driver unifies the per-dataset v5 evaluation scripts
(``experiments/eval_omniview_fusion_v5_h36m.py`` and
``experiments/eval_omniview_fusion_v5_mpiinf3dhp.py``) behind a single
YAML manifest, re-using the same split files already used by the training
pipeline.  It produces a single JSON/CSV report with frame-weighted
per-dataset and overall metrics.

Usage
-----
    # Dry-run: validate manifest and print commands without running inference
    python scripts/run_full_v5_benchmark_manifest.py \
        --manifest configs/benchmark_v25_smoke.yaml \
        --out outputs/benchmark_v25_smoke_dry \
        --dry-run

    # Real run (single sequence per dataset, smoke or real data)
    python scripts/run_full_v5_benchmark_manifest.py \
        --manifest configs/benchmark_v25_smoke.yaml \
        --out outputs/benchmark_v25_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml


METRIC_KEYS = (
    "mpjpe",
    "pa_mpjpe",
    "root_rel_mpjpe",
    "velocity_mpjpe",
    "bone_length_error",
    "pck@50mm",
    "pck@100mm",
    "pck@150mm",
    "pck_auc",
    "visibility_accuracy",
)


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _expand_split(yaml_path: Path, split: str) -> List[Dict[str, str]]:
    """Return [{'name': ..., 'file': ...}] for a split YAML."""
    manifest = _load_yaml(yaml_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Split YAML {yaml_path} must be a dictionary")
    files = manifest.get(split, [])
    if not files:
        raise ValueError(f"Split '{split}' not found or empty in {yaml_path}")
    return [
        {"name": Path(f).stem, "file": str(Path(f))}
        for f in files
    ]


def _frame_count(npz_path: Optional[str]) -> int:
    """Return number of frames in an .npz, defaulting to 1 if unknown."""
    if not npz_path:
        return 1
    try:
        data = np.load(npz_path)
        return int(data["joints_3d"].shape[0])
    except Exception:
        return 1


def _args_to_cli(common_args: Dict[str, Any]) -> List[str]:
    """Convert a dictionary of model/eval args to command-line flags."""
    cli: List[str] = []
    for key, value in common_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            for v in value:
                cli.extend([f"--{key}", str(v)])
        else:
            cli.extend([f"--{key}", str(value)])
    return cli


def _build_eval_command(
    checkpoint: str,
    common_args: Dict[str, Any],
    dataset: Dict[str, Any],
    sequence: Optional[Dict[str, str]],
    out_json: Path,
    out_csv: Path,
) -> List[str]:
    cmd = [sys.executable, dataset["script"], "--checkpoint", checkpoint]
    cmd.extend(_args_to_cli(common_args))

    if dataset.get("smoke"):
        cmd.append("--smoke")
    else:
        if sequence is None:
            raise ValueError("Non-smoke datasets require a sequence entry")
        cmd.extend(["--dataset", sequence["file"]])

    if dataset.get("run_robustness"):
        cmd.append("--run_robustness")
    if dataset.get("run_variable_views"):
        cmd.append("--run_variable_views")

    cmd.extend(["--out_json", str(out_json), "--out_csv", str(out_csv)])
    return cmd


def _average_metrics(
    reports: Sequence[Dict[str, Any]],
    weights: Sequence[float],
) -> Dict[str, Any]:
    """Frame-weighted average of a list of metric reports."""
    if not reports:
        return {}

    total = sum(weights)
    if total == 0:
        return reports[0] if reports else {}

    result: Dict[str, Any] = {}
    keys = [k for k in reports[0].keys() if not k.endswith("_per_joint")]
    scalar_keys = [k for k in keys if isinstance(reports[0][k], (int, float))]
    list_keys = [k for k in keys if isinstance(reports[0][k], list)]

    for k in scalar_keys:
        result[k] = float(
            sum(r[k] * w for r, w in zip(reports, weights)) / total
        )

    for k in list_keys:
        arrays = [np.asarray(r[k]) * w for r, w in zip(reports, weights)]
        if arrays:
            stacked = np.stack(arrays, axis=0)
            result[k] = (stacked.sum(axis=0) / total).tolist()

    return result


def _run_sequence(
    checkpoint: str,
    common_args: Dict[str, Any],
    dataset: Dict[str, Any],
    sequence: Optional[Dict[str, str]],
    out_json: Path,
    out_csv: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    cmd = _build_eval_command(checkpoint, common_args, dataset, sequence, out_json, out_csv)

    entry: Dict[str, Any] = {
        "name": sequence["name"] if sequence else dataset["name"],
        "command": cmd,
        "status": "dry_run" if dry_run else "pending",
    }

    if dry_run:
        entry["metrics"] = {}
        return entry

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        entry["status"] = "failed"
        entry["stdout"] = result.stdout
        entry["stderr"] = result.stderr
        entry["metrics"] = {}
        return entry

    with open(out_json) as f:
        eval_results = json.load(f)

    clean_metrics = eval_results.get("clean", {})
    entry["status"] = "completed"
    entry["metrics"] = clean_metrics
    entry["file"] = sequence["file"] if sequence else None
    return entry


def _run_dataset(
    checkpoint: str,
    common_args: Dict[str, Any],
    dataset: Dict[str, Any],
    out_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    ds_name = dataset["name"]
    ds_entries: List[Dict[str, Any]] = []
    weights: List[int] = []

    if dataset.get("smoke"):
        sequences: List[Optional[Dict[str, str]]] = [None]
    else:
        split_yaml = Path(dataset["split_yaml"])
        sequences = _expand_split(split_yaml, dataset.get("split", "test"))
        limit = dataset.get("limit")
        if limit is not None:
            sequences = sequences[: int(limit)]

    for seq in sequences:
        if dataset.get("smoke"):
            seq_name = "smoke"
        else:
            seq_name = seq["name"]  # type: ignore[index]
        out_json = out_dir / f"{ds_name}__{seq_name}.json"
        out_csv = out_dir / f"{ds_name}__{seq_name}.csv"
        entry = _run_sequence(
            checkpoint,
            common_args,
            dataset,
            seq,
            out_json,
            out_csv,
            dry_run,
        )
        ds_entries.append(entry)
        if entry["status"] in {"completed", "dry_run"}:
            if dataset.get("smoke"):
                weight = common_args.get("smoke_n_frames", 60)
            else:
                weight = _frame_count(str(entry.get("file", "")))
            weights.append(weight)

    metrics = [e["metrics"] for e in ds_entries if e["metrics"]]
    aggregated = _average_metrics(metrics, weights) if metrics else {}

    return {
        "name": ds_name,
        "script": dataset["script"],
        "sequences": ds_entries,
        "metrics": aggregated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full v5 benchmark from YAML manifest")
    parser.add_argument("--manifest", type=str, required=True, help="YAML manifest path")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands, skip model inference")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_yaml(Path(args.manifest))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest YAML must contain a top-level dictionary")

    checkpoint = manifest["checkpoint"]
    if not Path(checkpoint).exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    common_args = manifest.get("common_args", {})
    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets listed in manifest")

    dataset_results: List[Dict[str, Any]] = []
    all_reports: List[Dict[str, Any]] = []
    all_weights: List[int] = []

    for dataset in datasets:
        ds_result = _run_dataset(checkpoint, common_args, dataset, out_dir, args.dry_run)
        dataset_results.append(ds_result)
        if ds_result["metrics"]:
            # Use a weight of 1 for smoke datasets, otherwise sum of per-sequence weights.
            weight = 1 if dataset.get("smoke") else sum(
                _frame_count(str(e.get("file", ""))) for e in ds_result["sequences"]
                if e["status"] in {"completed", "dry_run"}
            )
            all_reports.append(ds_result["metrics"])
            all_weights.append(weight)

    overall = _average_metrics(all_reports, all_weights)

    summary = {
        "per_dataset": {d["name"]: d["metrics"] for d in dataset_results},
        "overall": overall,
    }

    final_report = {
        "manifest": args.manifest,
        "dry_run": args.dry_run,
        "checkpoint": checkpoint,
        "common_args": common_args,
        "datasets": dataset_results,
        "summary": summary,
    }

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)

    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "metric", "value"])
        for ds_name, metrics in summary["per_dataset"].items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.endswith("_per_joint"):
                    writer.writerow([ds_name, k, f"{v:.4f}"])
        writer.writerow(["", "", ""])
        writer.writerow(["overall", "", ""])
        for k, v in overall.items():
            if isinstance(v, (int, float)) and not k.endswith("_per_joint"):
                writer.writerow(["overall", k, f"{v:.4f}"])

    print(f"Benchmark report written to {json_path}")
    if args.dry_run:
        print("Dry-run complete; no model inference was performed.")
    else:
        for ds in dataset_results:
            mpjpe = ds["metrics"].get("mpjpe")
            pa = ds["metrics"].get("pa_mpjpe")
            if mpjpe is not None and pa is not None:
                print(f"  {ds['name']}: MPJPE={mpjpe:.2f} mm, PA-MPJPE={pa:.2f} mm")
            else:
                print(f"  {ds['name']}: metrics unavailable (see per-sequence logs)")


if __name__ == "__main__":
    main()
