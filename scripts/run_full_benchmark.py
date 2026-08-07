#!/usr/bin/env python3
"""Run a trained checkpoint across H36M, MPI-INF-3DHP and WebBridge test sets.

Usage:
    python scripts/run_full_benchmark.py \
        --manifest configs/benchmark_icra_cvpr_2027.yaml \
        --out outputs/icra_cvpr_2027_full_benchmark

Dry-run (no model inference):
    python scripts/run_full_benchmark.py \
        --manifest configs/benchmark_icra_cvpr_2027.yaml \
        --out outputs/dry_run \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


REQUIRED_MODEL_KEYS = {"model", "checkpoint", "clip_len", "d", "residual_hidden"}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _expand_dataset(name: str, path: str, split: Optional[str]) -> List[Dict[str, Any]]:
    """Return a list of {'name', 'file'} entries from a YAML split or a single .npz."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if p.suffix in {".yaml", ".yml"}:
        manifest = _load_yaml(p)
        if not isinstance(manifest, dict):
            raise ValueError(f"YAML split {path} must contain a top-level dictionary")
        files = manifest.get(split or "test", [])
        if not files:
            files = manifest.get("test", []) or []
        return [{"name": f"{name}__{Path(f).stem}", "file": str(Path(f))} for f in files]

    if p.suffix == ".npz":
        return [{"name": name, "file": str(p)}]

    raise ValueError(f"Unsupported dataset path: {path}")


def _frame_count(npz_path: str) -> int:
    try:
        data = np.load(npz_path)
        return int(data["joints_3d"].shape[0])
    except Exception:
        return 1


def _build_eval_command(model_config: Dict[str, Any], npz_path: str, output_json: Path) -> List[str]:
    cmd = [
        sys.executable,
        "experiments/eval_full_metrics.py",
        "--model", str(model_config["model"]),
        "--dataset", npz_path,
        "--checkpoint", str(model_config["checkpoint"]),
        "--clip_len", str(model_config["clip_len"]),
        "--d", str(model_config["d"]),
        "--residual_hidden", str(model_config["residual_hidden"]),
        "--batch_size", str(model_config.get("batch_size", 8)),
        "--gt_scale", str(model_config.get("gt_scale", 1.0)),
        "--camera_scale", str(model_config.get("camera_scale", 1.0)),
        "--val_stride", str(model_config.get("val_stride", 1)),
        "--output_json", str(output_json),
    ]

    if "source_n_views" in model_config:
        cmd.extend(["--source_n_views", str(model_config["source_n_views"])])

    if model_config["model"] in {
        "crossview_residual",
        "crossview_residual_pp",
        "crossview_residual_pp_visibility",
        "dynamic_gate_pp",
        "graph_skeleton_residual_pp",
        "epipolar_pp",
        "splat_pp",
        "kinematic_chain_pp",
        "bayesian_tri_pp",
        "bayesian_tri_v2_pp",
        "bayesian_tri_v2_visibility_pp",
        "bayesian_tri_v2_attention_entropy_pp",
        "epipolar_bias_v2_pp",
        "camera_conditioned_pp",
        "hierarchical_view_temporal_joint_pp",
    }:
        cmd.extend(["--n_st_layers", str(model_config.get("n_st_layers", 2))])
    else:
        cmd.extend(["--n_temporal_layers", str(model_config.get("n_temporal_layers", 2))])

    for key in (
        "graph_layers",
        "k",
        "target_k",
        "min_views",
        "parents",
        "symmetry_pairs",
        "n_view_layers",
        "n_view_groups",
        "n_joint_graph_layers",
    ):
        if key in model_config:
            cmd.extend([f"--{key}", str(model_config[key])])

    return cmd


def _make_placeholder_metrics() -> Dict[str, Any]:
    return {
        "mpjpe": 0.0,
        "pa_mpjpe": 0.0,
        "root_rel_mpjpe": 0.0,
        "velocity_mpjpe": 0.0,
        "bone_length_error": 0.0,
        "pck@50mm": 0.0,
        "pck@100mm": 0.0,
        "pck@150mm": 0.0,
        "pck_auc": 0.0,
        "per_joint_mpjpe": [],
        "per_joint_pa_mpjpe": [],
    }


def _average_sequence_metrics(
    sequence_reports: List[Dict[str, Any]], weights: List[int]
) -> Dict[str, Any]:
    if not sequence_reports:
        return _make_placeholder_metrics()

    total_weight = sum(weights)
    if total_weight == 0:
        return _make_placeholder_metrics()

    result: Dict[str, Any] = {}
    keys = [k for k in sequence_reports[0].keys() if not k.startswith("pck@") or "_per_joint" in k]
    # We average scalar keys and list keys separately.
    scalar_keys = [k for k in keys if isinstance(sequence_reports[0][k], (int, float))]
    for k in scalar_keys:
        result[k] = float(
            sum(r[k] * w for r, w in zip(sequence_reports, weights))
            / total_weight
        )

    list_keys = [k for k in keys if isinstance(sequence_reports[0][k], list)]
    for k in list_keys:
        arrays = [np.asarray(r[k]) * w for r, w in zip(sequence_reports, weights)]
        if arrays:
            stacked = np.stack(arrays, axis=0)
            result[k] = (stacked.sum(axis=0) / total_weight).tolist()

    # Keep original PCK thresholds as scalar values too.
    for k in sequence_reports[0]:
        if k.startswith("pck@") and "_per_joint" not in k:
            result[k] = float(
                sum(r[k] * w for r, w in zip(sequence_reports, weights))
                / total_weight
            )

    return result


def _run_eval(
    model_config: Dict[str, Any],
    sequence: Dict[str, Any],
    out_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    output_json = out_dir / f"{sequence['name']}.json"
    cmd = _build_eval_command(model_config, sequence["file"], output_json)

    if dry_run:
        return {
            "name": sequence["name"],
            "file": sequence["file"],
            "command": cmd,
            "metrics": _make_placeholder_metrics(),
            "status": "dry_run",
        }

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "name": sequence["name"],
            "file": sequence["file"],
            "command": cmd,
            "metrics": _make_placeholder_metrics(),
            "status": "failed",
            "stderr": result.stderr,
            "stdout": result.stdout,
        }

    with open(output_json) as f:
        metrics = json.load(f)

    return {
        "name": sequence["name"],
        "file": sequence["file"],
        "command": cmd,
        "metrics": metrics,
        "status": "completed",
    }


def main():
    parser = argparse.ArgumentParser(description="Full cross-dataset benchmark")
    parser.add_argument("--manifest", type=str, required=True, help="YAML manifest path")
    parser.add_argument("--out", type=str, required=True, help="Output directory prefix")
    parser.add_argument("--dry-run", action="store_true", help="Skip model inference and emit placeholder metrics")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_yaml(Path(args.manifest))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest YAML must contain a top-level dictionary")

    model_config = manifest.get("model_config", {})
    missing = REQUIRED_MODEL_KEYS - set(model_config.keys())
    if missing:
        raise ValueError(f"model_config missing required keys: {missing}")

    if not Path(model_config["checkpoint"]).exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint not found: {model_config['checkpoint']}")

    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets listed in manifest")

    dataset_results: List[Dict[str, Any]] = []
    all_sequence_reports: List[Dict[str, Any]] = []
    all_weights: List[int] = []

    for ds in datasets:
        ds_name = ds["name"]
        ds_path = ds["path"]
        split = ds.get("split", "test")
        sequences = _expand_dataset(ds_name, ds_path, split)

        per_dataset: List[Dict[str, Any]] = []
        weights: List[int] = []
        for seq in sequences:
            report = _run_eval(model_config, seq, out_dir, args.dry_run)
            per_dataset.append(report)
            if report["status"] in {"completed", "dry_run"}:
                weights.append(_frame_count(report["file"]))
                all_weights.append(weights[-1])
                all_sequence_reports.append(report["metrics"])

        if per_dataset:
            seq_metrics = [r["metrics"] for r in per_dataset]
            aggregated = _average_sequence_metrics(seq_metrics, weights)
        else:
            aggregated = _make_placeholder_metrics()

        dataset_results.append({
            "name": ds_name,
            "path": ds_path,
            "split": split,
            "sequences": per_dataset,
            "metrics": aggregated,
        })

    overall = _average_sequence_metrics(all_sequence_reports, all_weights)

    summary = {
        "per_dataset": {d["name"]: d["metrics"] for d in dataset_results},
        "overall": overall,
    }

    final_report = {
        "manifest": args.manifest,
        "dry_run": args.dry_run,
        "model_config": model_config,
        "datasets": dataset_results,
        "summary": summary,
    }

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"Benchmark report written to {json_path}")
    if args.dry_run:
        print("Dry-run complete; no model inference was performed.")
    else:
        for ds in dataset_results:
            mpjpe = ds["metrics"].get("mpjpe")
            pa = ds["metrics"].get("pa_mpjpe")
            print(f"  {ds['name']}: MPJPE={mpjpe:.2f} mm, PA-MPJPE={pa:.2f} mm")


if __name__ == "__main__":
    main()
