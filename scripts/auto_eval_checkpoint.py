#!/usr/bin/env python3
"""Auto-evaluate a trained checkpoint on the standard H36M / MPI / WebBridge test sets.

This script wraps ``experiments/eval_full_metrics.py`` and runs a single trained
``.pth`` over the canonical test sequences, then aggregates per-dataset and
overall metrics into a JSON report.

Usage
-----
    python scripts/auto_eval_checkpoint.py \
        --checkpoint outputs/v23_kap_no_ba.pth \
        --model crossview_residual_pp \
        --clip_len 13 --d 64 --n_st_layers 2 --residual_hidden 128 \
        --source_n_views 14 \
        --out outputs/v23_auto_eval

Dry-run (print commands, do not run inference):
    python scripts/auto_eval_checkpoint.py \
        --checkpoint outputs/v23_kap_no_ba.pth \
        --model crossview_residual_pp \
        --out outputs/v23_auto_eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SCRIPT = ROOT / "experiments" / "eval_full_metrics.py"

DEFAULT_DATA_ROOT = ROOT / "data" / "webbridge"


# Canonical test sequences used for the three benchmarks.
# Keys are glob patterns relative to data/webbridge.
DEFAULT_TEST_SETS: Dict[str, List[str]] = {
    "h36m": ["h36m_meters/s_11_acts_*_multiview_m.npz"],
    "mpi": ["mpi_inf_3dhp/test_set/TS*_v14_multiview.npz"],
    "webbridge_3dpw": ["3dpw/converted/test/*.npz"],
}


def _glob_sequences(patterns: Sequence[str]) -> List[Path]:
    """Expand a list of glob patterns, sorted, with duplicates removed."""
    seen: set = set()
    results: List[Path] = []
    for pat in patterns:
        for p in sorted(DEFAULT_DATA_ROOT.glob(pat)):
            if p not in seen:
                seen.add(p)
                results.append(p)
    return results


def _frame_count(npz_path: Path) -> int:
    try:
        data = np.load(npz_path)
        return int(data["joints_3d"].shape[0])
    except Exception:
        return 1


def _build_eval_command(
    args: argparse.Namespace,
    npz_path: Path,
    output_json: Path,
) -> List[str]:
    """Build the command list for ``experiments/eval_full_metrics.py``."""
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model", args.model,
        "--dataset", str(npz_path),
        "--checkpoint", str(args.checkpoint),
        "--clip_len", str(args.clip_len),
        "--d", str(args.d),
        "--residual_hidden", str(args.residual_hidden),
        "--batch_size", str(args.batch_size),
        "--gt_scale", str(args.gt_scale),
        "--camera_scale", str(args.camera_scale),
        "--val_stride", str(args.val_stride),
        "--output_json", str(output_json),
    ]

    # Cross-view / spatio-temporal models use --n_st_layers; others use
    # --n_temporal_layers.  This list must stay in sync with
    # experiments/eval_full_metrics.py.
    if args.model in {
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
        cmd.extend(["--n_st_layers", str(args.n_st_layers)])
    else:
        cmd.extend(["--n_temporal_layers", str(args.n_temporal_layers)])

    if args.source_n_views is not None:
        cmd.extend(["--source_n_views", str(args.source_n_views)])

    # Pass through optional graph / hierarchical parameters when provided.
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
        val = getattr(args, key, None)
        if val is not None:
            cmd.extend([f"--{key}", str(val)])

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
    scalar_keys = [
        k for k in sequence_reports[0].keys()
        if isinstance(sequence_reports[0][k], (int, float)) and not k.startswith("pck@")
    ]
    for k in scalar_keys:
        result[k] = float(
            sum(r[k] * w for r, w in zip(sequence_reports, weights)) / total_weight
        )

    list_keys = [
        k for k in sequence_reports[0].keys()
        if isinstance(sequence_reports[0][k], list)
    ]
    for k in list_keys:
        arrays = [np.asarray(r[k]) * w for r, w in zip(sequence_reports, weights)]
        if arrays:
            stacked = np.stack(arrays, axis=0)
            result[k] = (stacked.sum(axis=0) / total_weight).tolist()

    # Keep PCK thresholds as scalar values.
    for k in sequence_reports[0]:
        if k.startswith("pck@") and "_per_joint" not in k:
            result[k] = float(
                sum(r[k] * w for r, w in zip(sequence_reports, weights)) / total_weight
            )

    return result


def _run_eval(
    args: argparse.Namespace,
    npz_path: Path,
    out_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    output_json = out_dir / f"{npz_path.stem}.json"
    cmd = _build_eval_command(args, npz_path, output_json)

    if dry_run:
        return {
            "name": npz_path.stem,
            "file": str(npz_path),
            "command": cmd,
            "metrics": _make_placeholder_metrics(),
            "status": "dry_run",
        }

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "name": npz_path.stem,
            "file": str(npz_path),
            "command": cmd,
            "metrics": _make_placeholder_metrics(),
            "status": "failed",
            "stderr": result.stderr,
            "stdout": result.stdout,
        }

    with open(output_json) as f:
        metrics = json.load(f)

    return {
        "name": npz_path.stem,
        "file": str(npz_path),
        "command": cmd,
        "metrics": metrics,
        "status": "completed",
    }


def _collect_test_sequences(
    selected: Optional[List[str]] = None,
    h36m: Optional[List[str]] = None,
    mpi: Optional[List[str]] = None,
    webbridge: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """Collect test sequence paths per dataset.

    When no overrides are provided, the canonical test splits are used.
    Each argument accepts glob patterns relative to ``data/webbridge``.
    """
    selected = selected or list(DEFAULT_TEST_SETS.keys())
    test_sets: Dict[str, List[str]] = {}
    overrides: Dict[str, Optional[List[str]]] = {
        "h36m": h36m,
        "mpi": mpi,
        "webbridge_3dpw": webbridge,
    }

    for name in selected:
        override = overrides.get(name)
        test_sets[name] = override if override is not None else DEFAULT_TEST_SETS[name]

    return {name: _glob_sequences(patterns) for name, patterns in test_sets.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Auto-evaluate a trained .pth on H36M / MPI / WebBridge test sets."
    )
    # Checkpoint & model configuration.
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the trained .pth checkpoint.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture name (see experiments/eval_full_metrics.py).")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2)
    parser.add_argument("--n_joint_graph_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--source_n_views", type=int, default=None,
                        help="Fixed view count of the trained model (for variable-view inference).")
    parser.add_argument("--parents", type=str, default=None)
    parser.add_argument("--symmetry_pairs", type=str, default=None)

    # Evaluation behaviour.
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)

    # I/O.
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory for per-sequence JSONs and final report.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands and emit placeholder metrics without running the model.")

    parser.add_argument("--datasets", nargs="+",
                        choices=list(DEFAULT_TEST_SETS.keys()),
                        default=list(DEFAULT_TEST_SETS.keys()),
                        help="Which test sets to evaluate (default: all three).")

    # Optional test-set overrides (default: auto-discover canonical splits).
    parser.add_argument("--h36m-patterns", nargs="+", default=None,
                        help="Glob pattern(s) for H36M test sequences relative to data/webbridge.")
    parser.add_argument("--mpi-patterns", nargs="+", default=None,
                        help="Glob pattern(s) for MPI test sequences relative to data/webbridge.")
    parser.add_argument("--webbridge-patterns", nargs="+", default=None,
                        help="Glob pattern(s) for WebBridge test sequences relative to data/webbridge.")

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Fix the --model choices: the placeholder above is a stub; replace with the
    # real model registry used by experiments/eval_full_metrics.py.
    # We re-parse would not be safe, so we validate directly here.
    from experiments.eval_full_metrics import MODEL_CLASSES
    if args.model not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(MODEL_CLASSES.keys())}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    test_sequences = _collect_test_sequences(
        selected=args.datasets,
        h36m=args.h36m_patterns,
        mpi=args.mpi_patterns,
        webbridge=args.webbridge_patterns,
    )

    dataset_results: List[Dict[str, Any]] = []
    all_sequence_reports: List[Dict[str, Any]] = []
    all_weights: List[int] = []

    for ds_name, sequences in test_sequences.items():
        per_dataset: List[Dict[str, Any]] = []
        weights: List[int] = []
        for seq_path in sequences:
            report = _run_eval(args, seq_path, out_dir, args.dry_run)
            per_dataset.append(report)
            if report["status"] in {"completed", "dry_run"}:
                weights.append(_frame_count(seq_path))
                all_weights.append(weights[-1])
                all_sequence_reports.append(report["metrics"])

        if per_dataset:
            seq_metrics = [r["metrics"] for r in per_dataset]
            aggregated = _average_sequence_metrics(seq_metrics, weights)
        else:
            aggregated = _make_placeholder_metrics()

        dataset_results.append({
            "name": ds_name,
            "sequences": per_dataset,
            "metrics": aggregated,
        })

    overall = _average_sequence_metrics(all_sequence_reports, all_weights)

    summary = {
        "per_dataset": {d["name"]: d["metrics"] for d in dataset_results},
        "overall": overall,
    }

    def _jsonify(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonify(v) for v in value]
        return value

    final_report = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "dry_run": args.dry_run,
        "args": _jsonify(vars(args)),
        "datasets": dataset_results,
        "summary": summary,
    }

    json_path = out_dir / "auto_eval_results.json"
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"Auto-evaluation report written to {json_path}")
    if args.dry_run:
        print("Dry-run complete; no model inference was performed.")
    else:
        for ds in dataset_results:
            mpjpe = ds["metrics"].get("mpjpe")
            pa = ds["metrics"].get("pa_mpjpe")
            print(f"  {ds['name']}: MPJPE={mpjpe:.2f} mm, PA-MPJPE={pa:.2f} mm")


if __name__ == "__main__":
    main()
