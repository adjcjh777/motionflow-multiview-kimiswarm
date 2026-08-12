#!/usr/bin/env python3
"""Compare v85 sparse-view variable-view results against the v25 DLT-fallback baseline.

Reads CSV/JSON outputs produced by ``experiments/eval_variable_views.py`` and
generates a structured comparison (JSON/CSV/Markdown) plus an optional bar
chart.  Both single-file results and per-k split files (``*_k2.json``,
``*_k3.json``, ``*_k4.json``) are supported.

Usage
-----
    # Compare full v85 results against the v25 DLT-fallback baseline.
    python scripts/analyze_v85_sparse_view.py \
        --v85_json outputs/variable_view_v85_random_view_dropout_medium_a800.json \
        --baseline_json outputs/variable_view_v25_true_gt_stability_a800_dlt_fallback.json \
        --out_dir outputs/v85_sparse_view_analysis

    # Compare per-k split v85 files.
    python scripts/analyze_v85_sparse_view.py \
        --v85_json "outputs/variable_view_v85_random_view_dropout_medium_a800_k*.json" \
        --baseline_json outputs/variable_view_v25_true_gt_stability_a800_dlt_fallback.json \
        --out_dir outputs/v85_sparse_view_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _csv_rows_to_per_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert CSV rows from ``eval_variable_views.py`` to the JSON structure."""
    per_dataset: Dict[str, Any] = {}
    for row in rows:
        dataset = row["dataset"]
        k = row["k"]
        if dataset not in per_dataset:
            per_dataset[dataset] = {}
        if k in per_dataset[dataset]:
            raise ValueError(f"Duplicate (dataset={dataset}, k={k}) in CSV rows")
        per_dataset[dataset][k] = {
            "mpjpe_at_k": float(row["mpjpe_at_k"]),
            "mean_mm": float(row["mean_mm"]),
            "std_mm": float(row["std_mm"]),
            "n_subsets": int(row["n_subsets"]),
            "temporal_jerk": float(row["temporal_jerk"]),
        }
    return {"per_dataset": per_dataset}


def _resolve_paths(pattern: Optional[str]) -> List[str]:
    """Expand a path or glob pattern to a sorted list of existing files."""
    if pattern is None:
        return []
    # If the literal path exists, use it.  Otherwise treat as a glob.
    p = Path(pattern)
    if p.is_file():
        return [str(p)]
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return matches


def load_results(json_path: Optional[str], csv_path: Optional[str]) -> Dict[str, Any]:
    """Load variable-view results from JSON, CSV, or a set of per-k files."""
    if json_path is not None:
        json_files = _resolve_paths(json_path)
        if len(json_files) == 1:
            results = _load_json(json_files[0])
        else:
            results = _merge_per_k_json(json_files)
    elif csv_path is not None:
        csv_files = _resolve_paths(csv_path)
        if len(csv_files) == 1:
            results = _csv_rows_to_per_dataset(_load_csv(csv_files[0]))
        else:
            results = _merge_per_k_csv(csv_files)
    else:
        raise ValueError("Either --v85_json/--v85_csv or --baseline_json/--baseline_csv must be provided")

    if "per_dataset" not in results:
        raise ValueError("Input JSON/CSV must contain a 'per_dataset' top-level key")
    return results


def _merge_per_k_json(paths: List[str]) -> Dict[str, Any]:
    """Merge multiple per-k JSON files into a single ``per_dataset`` structure."""
    merged: Dict[str, Any] = {"per_dataset": {}}
    for path in paths:
        data = _load_json(path)
        if "per_dataset" not in data:
            raise ValueError(f"File {path} missing 'per_dataset' key")
        for dataset, k_map in data["per_dataset"].items():
            merged["per_dataset"].setdefault(dataset, {})
            for k, metrics in k_map.items():
                if k in merged["per_dataset"][dataset]:
                    raise ValueError(
                        f"Duplicate (dataset={dataset}, k={k}) from file {path}"
                    )
                merged["per_dataset"][dataset][k] = metrics
    return merged


def _merge_per_k_csv(paths: List[str]) -> Dict[str, Any]:
    """Merge multiple per-k CSV files into a single ``per_dataset`` structure."""
    all_rows: List[Dict[str, Any]] = []
    for path in paths:
        all_rows.extend(_load_csv(path))
    return _csv_rows_to_per_dataset(all_rows)


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def build_per_frame_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a per-frame summary from metrics containing ``per_frame`` arrays.

    The optional ``per_frame`` field is expected to be a list of frame-wise
    MPJPE values in millimetres.  When present, this function returns summary
    statistics (mean, std, min, max, best/worst frame index) for each
    ``(dataset, k)`` pair.
    """
    analysis: Dict[str, Any] = {}
    for dataset, k_map in results["per_dataset"].items():
        for k_str, metrics in k_map.items():
            per_frame = metrics.get("per_frame")
            if not isinstance(per_frame, (list, tuple)) or not per_frame:
                continue
            try:
                arr = [float(v) for v in per_frame]
            except (TypeError, ValueError):
                continue
            n = len(arr)
            if n == 0:
                continue
            mean = sum(arr) / n
            variance = sum((x - mean) ** 2 for x in arr) / n
            analysis.setdefault(dataset, {})[k_str] = {
                "mean_mm": mean,
                "std_mm": variance ** 0.5,
                "min_mm": min(arr),
                "max_mm": max(arr),
                "best_frame": int(arr.index(min(arr))),
                "worst_frame": int(arr.index(max(arr))),
                "n_frames": n,
            }
    return analysis


# ---------------------------------------------------------------------------
# Per-camera analysis
# ---------------------------------------------------------------------------

def build_per_camera_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build per-camera summaries from metrics containing ``subsets``/``per_subset``.

    For each camera index, reports the number of evaluated subsets in which it
    appears and the mean/min/max MPJPE across those subsets.
    """
    analysis: Dict[str, Any] = {}
    for dataset, k_map in results["per_dataset"].items():
        for k_str, metrics in k_map.items():
            subsets = metrics.get("subsets")
            per_subset = metrics.get("per_subset")
            if not isinstance(subsets, (list, tuple)) or not isinstance(per_subset, (list, tuple)):
                continue
            if len(subsets) != len(per_subset):
                continue
            camera_stats: Dict[int, Dict[str, Any]] = {}
            for subset, sub_metrics in zip(subsets, per_subset):
                if not isinstance(sub_metrics, dict):
                    continue
                mpjpe_val = sub_metrics.get("mpjpe")
                if mpjpe_val is None:
                    continue
                try:
                    mpjpe_val = float(mpjpe_val)
                except (TypeError, ValueError):
                    continue
                for cam in subset:
                    if cam not in camera_stats:
                        camera_stats[cam] = {
                            "count": 0,
                            "sum": 0.0,
                            "min": float("inf"),
                            "max": 0.0,
                        }
                    stats = camera_stats[cam]
                    stats["count"] += 1
                    stats["sum"] += mpjpe_val
                    if mpjpe_val < stats["min"]:
                        stats["min"] = mpjpe_val
                    if mpjpe_val > stats["max"]:
                        stats["max"] = mpjpe_val
            for cam, stats in camera_stats.items():
                count = stats["count"]
                camera_stats[cam] = {
                    "camera": cam,
                    "count": count,
                    "mean_mpjpe_mm": stats["sum"] / count if count > 0 else None,
                    "min_mpjpe_mm": stats["min"] if count > 0 else None,
                    "max_mpjpe_mm": stats["max"] if count > 0 else None,
                }
            if camera_stats:
                analysis.setdefault(dataset, {})[k_str] = {
                    "cameras": {str(c): camera_stats[c] for c in sorted(camera_stats)},
                }
    return analysis


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _iter_metrics(results: Dict[str, Any]) -> List[Tuple[str, int, Dict[str, Any]]]:
    """Yield (dataset, k, metrics) tuples sorted by dataset and k."""
    items: List[Tuple[str, int, Dict[str, Any]]] = []
    for dataset, k_map in results["per_dataset"].items():
        for k_str, metrics in k_map.items():
            items.append((dataset, int(k_str), metrics))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def _get_metric(metrics: Dict[str, Any], key: str) -> float:
    """Return a metric value, raising a clear error if missing."""
    if key not in metrics:
        raise KeyError(f"Metric '{key}' not found in results; available keys: {list(metrics.keys())}")
    return float(metrics[key])


def build_comparison(v85_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a dataset/k comparison between v85 and the baseline."""
    v85_items = _iter_metrics(v85_results)
    baseline_items = { (d, k): m for d, k, m in _iter_metrics(baseline_results) }

    per_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    combined_rows: List[Dict[str, Any]] = []

    for dataset, k, v85_metrics in v85_items:
        v85_mpjpe = _get_metric(v85_metrics, "mpjpe_at_k")
        baseline_metrics = baseline_items.get((dataset, k))
        row: Dict[str, Any] = {
            "dataset": dataset,
            "k": k,
            "v85_mpjpe_mm": v85_mpjpe,
        }
        if baseline_metrics is not None:
            baseline_mpjpe = _get_metric(baseline_metrics, "mpjpe_at_k")
            delta = v85_mpjpe - baseline_mpjpe
            # Improvement: positive means v85 is better than baseline.
            improvement = (baseline_mpjpe - v85_mpjpe) / baseline_mpjpe * 100.0
            row.update(
                {
                    "baseline_mpjpe_mm": baseline_mpjpe,
                    "delta_mm": delta,
                    "improvement_pct": improvement,
                }
            )
        else:
            row.update(
                {
                    "baseline_mpjpe_mm": None,
                    "delta_mm": None,
                    "improvement_pct": None,
                }
            )

        per_dataset.setdefault(dataset, {})[str(k)] = row
        combined_rows.append(row)

    # Combined S9/S11 weighted by n_subsets when available.
    combined: Dict[str, Dict[str, Any]] = {}
    for dataset, k, metrics in v85_items:
        k_str = str(k)
        if k_str not in combined:
            combined[k_str] = {
                "k": k,
                "v85_weighted_mpjpe_mm": 0.0,
                "baseline_weighted_mpjpe_mm": 0.0,
                "total_weight": 0,
            }
        weight = int(metrics.get("n_subsets", 1))
        combined[k_str]["v85_weighted_mpjpe_mm"] += _get_metric(metrics, "mpjpe_at_k") * weight
        baseline_metrics = baseline_items.get((dataset, k))
        if baseline_metrics is not None:
            combined[k_str]["baseline_weighted_mpjpe_mm"] += (
                _get_metric(baseline_metrics, "mpjpe_at_k") * weight
            )
        combined[k_str]["total_weight"] += weight

    for k_str, c in combined.items():
        w = c["total_weight"]
        if w > 0:
            c["v85_weighted_mpjpe_mm"] /= w
            c["baseline_weighted_mpjpe_mm"] /= w
        else:
            c["v85_weighted_mpjpe_mm"] = None
            c["baseline_weighted_mpjpe_mm"] = None

    return {
        "per_dataset": per_dataset,
        "combined": combined,
        "rows": combined_rows,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(comparison: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)


def write_csv(comparison: Dict[str, Any], path: str) -> None:
    rows = comparison["rows"]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_frame_csv(per_frame: Dict[str, Any], path: str) -> None:
    """Write per-frame summary to CSV with one row per (dataset, k)."""
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(per_frame):
        for k_str in sorted(per_frame[dataset], key=int):
            row = dict(per_frame[dataset][k_str])
            row["dataset"] = dataset
            row["k"] = int(k_str)
            rows.append(row)
    if not rows:
        return
    fieldnames = ["dataset", "k", "mean_mm", "std_mm", "min_mm", "max_mm",
                  "best_frame", "worst_frame", "n_frames"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_camera_csv(per_camera: Dict[str, Any], path: str) -> None:
    """Write per-camera summary to CSV with one row per (dataset, k, camera)."""
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(per_camera):
        for k_str in sorted(per_camera[dataset], key=int):
            cameras = per_camera[dataset][k_str].get("cameras", {})
            for cam in sorted(cameras, key=int):
                info = cameras[cam]
                rows.append({
                    "dataset": dataset,
                    "k": int(k_str),
                    "camera": info["camera"],
                    "count": info["count"],
                    "mean_mpjpe_mm": info["mean_mpjpe_mm"],
                    "min_mpjpe_mm": info["min_mpjpe_mm"],
                    "max_mpjpe_mm": info["max_mpjpe_mm"],
                })
    if not rows:
        return
    fieldnames = ["dataset", "k", "camera", "count", "mean_mpjpe_mm", "min_mpjpe_mm", "max_mpjpe_mm"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(comparison: Dict[str, Any], path: str, baseline_name: str = "v25 DLT-fallback",
                    per_frame: Optional[Dict[str, Any]] = None,
                    per_camera: Optional[Dict[str, Any]] = None) -> None:
    lines: List[str] = []
    lines.append("# v85 Sparse-View Analysis vs. Baseline\n")
    lines.append(f"Baseline: **{baseline_name}**\n\n")

    lines.append("## Per-Dataset Results\n")
    lines.append("| dataset | k | v85 (mm) | baseline (mm) | delta (mm) | improvement (%) |\n")
    lines.append("|---------|---|----------|---------------|-----------|------------------|\n")
    for row in comparison["rows"]:
        delta = row.get("delta_mm")
        improvement = row.get("improvement_pct")
        baseline = row.get("baseline_mpjpe_mm")

        baseline_str = f"{baseline:13.2f}" if baseline is not None else "N/A".rjust(13)
        delta_str = f"{delta:9.2f}" if delta is not None else "N/A".rjust(9)
        improvement_str = f"{improvement:16.2f}" if improvement is not None else "N/A".rjust(16)

        lines.append(
            f"| {row['dataset']:7s} | {row['k']} | "
            f"{row['v85_mpjpe_mm']:8.2f} | "
            f"{baseline_str} | "
            f"{delta_str} | "
            f"{improvement_str} |\n"
        )

    lines.append("\n## Combined (S9 + S11) Weighted Average\n")
    lines.append("| k | v85 (mm) | baseline (mm) |\n")
    lines.append("|---|----------|---------------|\n")
    for k_str in sorted(comparison["combined"], key=int):
        c = comparison["combined"][k_str]
        v85_val = c["v85_weighted_mpjpe_mm"]
        baseline_val = c["baseline_weighted_mpjpe_mm"]
        v85_str = f"{v85_val:.2f}" if v85_val is not None else "N/A"
        baseline_str = f"{baseline_val:.2f}" if baseline_val is not None else "N/A"
        lines.append(f"| {k_str} | {v85_str:8s} | {baseline_str:13s} |\n")

    if per_frame:
        lines.append("\n## Per-Frame Analysis\n")
        lines.append("| dataset | k | mean (mm) | std (mm) | min (mm) | max (mm) | best frame | worst frame |\n")
        lines.append("|---------|---|-----------|----------|----------|----------|------------|-------------|\n")
        for dataset in sorted(per_frame):
            for k_str in sorted(per_frame[dataset], key=int):
                stats = per_frame[dataset][k_str]
                lines.append(
                    f"| {dataset:7s} | {k_str} | "
                    f"{stats['mean_mm']:9.2f} | "
                    f"{stats['std_mm']:8.2f} | "
                    f"{stats['min_mm']:8.2f} | "
                    f"{stats['max_mm']:8.2f} | "
                    f"{stats['best_frame']:10d} | "
                    f"{stats['worst_frame']:11d} |\n"
                )

    if per_camera:
        lines.append("\n## Per-Camera Analysis\n")
        lines.append("| dataset | k | camera | count | mean (mm) | min (mm) | max (mm) |\n")
        lines.append("|---------|---|--------|-------|-----------|----------|----------|\n")
        for dataset in sorted(per_camera):
            for k_str in sorted(per_camera[dataset], key=int):
                cameras = per_camera[dataset][k_str].get("cameras", {})
                for cam in sorted(cameras, key=int):
                    info = cameras[cam]
                    lines.append(
                        f"| {dataset:7s} | {k_str} | {info['camera']:6d} | "
                        f"{info['count']:5d} | "
                        f"{info['mean_mpjpe_mm']:9.2f} | "
                        f"{info['min_mpjpe_mm']:8.2f} | "
                        f"{info['max_mpjpe_mm']:8.2f} |\n"
                    )

    with open(path, "w") as f:
        f.writelines(lines)


def write_plot(comparison: Dict[str, Any], path: str, baseline_name: str = "v25 DLT-fallback") -> None:
    """Draw a grouped bar chart of v85 vs baseline MPJPE per dataset and k."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install it or use --no_plot.")

    rows = comparison["rows"]
    if not rows:
        return

    # Group rows by dataset.
    datasets: Dict[str, List[Tuple[int, float, float]]] = {}
    for row in rows:
        if row.get("baseline_mpjpe_mm") is None:
            continue
        datasets.setdefault(row["dataset"], []).append(
            (row["k"], row["v85_mpjpe_mm"], row["baseline_mpjpe_mm"])
        )

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
    axes = axes.flatten()

    for ax, (dataset, values) in zip(axes, sorted(datasets.items())):
        values = sorted(values, key=lambda x: x[0])
        ks = [v[0] for v in values]
        v85_vals = [v[1] for v in values]
        baseline_vals = [v[2] for v in values]

        x = range(len(ks))
        width = 0.35
        ax.bar([i - width / 2 for i in x], v85_vals, width, label="v85")
        ax.bar([i + width / 2 for i in x], baseline_vals, width, label=baseline_name)
        ax.set_xlabel("k (active views)")
        ax.set_ylabel("MPJPE (mm)")
        ax.set_title(f"{dataset} sparse-view comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v85 sparse-view variable-view results against the v25 DLT-fallback baseline.",
    )
    parser.add_argument("--v85_json", type=str, default=None,
                        help="Path or glob to v85 JSON result(s)")
    parser.add_argument("--baseline_json", type=str, default=None,
                        help="Path to v25 DLT-fallback JSON result")
    parser.add_argument("--v85_csv", type=str, default=None,
                        help="Path or glob to v85 CSV result(s) (optional)")
    parser.add_argument("--baseline_csv", type=str, default=None,
                        help="Path to v25 DLT-fallback CSV result (optional)")
    parser.add_argument("--out_dir", type=str, default="outputs/v85_sparse_view_analysis",
                        help="Directory for output files")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path for comparison JSON (default: <out_dir>/comparison.json)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path for comparison CSV (default: <out_dir>/comparison.csv)")
    parser.add_argument("--output_md", type=str, default=None,
                        help="Path for Markdown report (default: <out_dir>/report.md)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--baseline_name", type=str, default="v25 DLT-fallback",
                        help="Display name for the baseline in reports")
    parser.add_argument("--per_frame", action="store_true",
                        help="Compute per-frame MPJPE statistics from any per-frame arrays")
    parser.add_argument("--per_camera_analysis", action="store_true",
                        help="Compute per-camera MPJPE statistics from any subset/per-subset data")
    args = parser.parse_args()

    if not args.v85_json and not args.v85_csv:
        parser.error("Either --v85_json or --v85_csv must be provided")
    if not args.baseline_json and not args.baseline_csv:
        parser.error("Either --baseline_json or --baseline_csv must be provided")

    return args


def main() -> None:
    args = parse_args()

    v85_results = load_results(args.v85_json, args.v85_csv)
    baseline_results = load_results(args.baseline_json, args.baseline_csv)
    comparison = build_comparison(v85_results, baseline_results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_json or str(out_dir / "comparison.json")
    csv_path = args.output_csv or str(out_dir / "comparison.csv")
    md_path = args.output_md or str(out_dir / "report.md")

    per_frame: Optional[Dict[str, Any]] = None
    per_camera: Optional[Dict[str, Any]] = None

    if args.per_frame:
        per_frame = build_per_frame_analysis(v85_results)
        if per_frame:
            write_json(per_frame, str(out_dir / "per_frame.json"))
            _write_per_frame_csv(per_frame, str(out_dir / "per_frame.csv"))
            print(f"Per-frame JSON saved to: {out_dir / 'per_frame.json'}")
            print(f"Per-frame CSV saved to:  {out_dir / 'per_frame.csv'}")

    if args.per_camera_analysis:
        per_camera = build_per_camera_analysis(v85_results)
        if per_camera:
            write_json(per_camera, str(out_dir / "per_camera.json"))
            _write_per_camera_csv(per_camera, str(out_dir / "per_camera.csv"))
            print(f"Per-camera JSON saved to: {out_dir / 'per_camera.json'}")
            print(f"Per-camera CSV saved to:  {out_dir / 'per_camera.csv'}")

    write_json(comparison, json_path)
    write_csv(comparison, csv_path)
    write_markdown(comparison, md_path, baseline_name=args.baseline_name,
                    per_frame=per_frame, per_camera=per_camera)

    if not args.no_plot:
        plot_path = out_dir / "comparison.png"
        write_plot(comparison, str(plot_path), baseline_name=args.baseline_name)
        print(f"Plot saved to: {plot_path}")

    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to:  {csv_path}")
    print(f"Report saved to: {md_path}")

    # Print a concise summary to stdout.
    print("\n" + "=" * 60)
    print("v85 vs baseline summary:")
    for row in comparison["rows"]:
        delta = row.get("delta_mm")
        improvement = row.get("improvement_pct")
        delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
        improvement_str = f"{improvement:+.2f}%" if improvement is not None else "N/A"
        print(
            f"  {row['dataset']} k={row['k']}: v85={row['v85_mpjpe_mm']:.2f} mm, "
            f"baseline={row.get('baseline_mpjpe_mm', 'N/A')} mm, "
            f"delta={delta_str}, improvement={improvement_str}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
