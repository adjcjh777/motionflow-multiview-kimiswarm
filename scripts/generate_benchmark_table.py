#!/usr/bin/env python3
"""Generate a Markdown/CSV benchmark table from result JSON files.

Supports the JSON layouts produced by the repository's benchmark and
auto-evaluation scripts:

* WebBridge benchmark manifest (``outputs/webbridge_benchmark_*.json``)
* Auto-eval report (``outputs/*/auto_eval_results.json``)
* Dry-run benchmark manifest (``outputs/benchmark_dry/benchmark_results.json``)
* Generic per-condition metrics JSON

Usage
-----
    python scripts/generate_benchmark_table.py \\
        --inputs outputs/webbridge_benchmark_crossview_residual_smoke_v2.json \\
        --output docs/benchmark_table.md

    python scripts/generate_benchmark_table.py \\
        --inputs outputs/benchmark_dry/benchmark_results.json \\
        --metrics mpjpe pa_mpjpe pck_50 pck_100 \\
        --csv outputs/benchmark_dry/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Map canonical metric names to display labels.
METRIC_LABELS: Dict[str, str] = {
    "mpjpe": "MPJPE (mm)",
    "mpjpe_mm": "MPJPE (mm)",
    "pa_mpjpe": "PA-MPJPE (mm)",
    "pa_mpjpe_mm": "PA-MPJPE (mm)",
    "root_rel_mpjpe": "Root-rel MPJPE (mm)",
    "velocity_mpjpe": "Velocity MPJPE (mm)",
    "bone_length_error": "Bone-length err (mm)",
    "pck_50": "PCK@50",
    "pck_100": "PCK@100",
    "pck_150": "PCK@150",
    "pck_auc": "PCK AUC",
    "pck@50mm": "PCK@50",
    "pck@100mm": "PCK@100",
    "pck@150mm": "PCK@150",
    "pck_auc": "PCK AUC",
}

# Default column order when the user does not specify --metrics.
DEFAULT_METRICS: List[str] = [
    "mpjpe",
    "pa_mpjpe",
    "pck_50",
    "pck_100",
    "pck_150",
    "pck_auc",
]


def _norm_metric_key(key: str) -> str:
    """Normalize a metric key to the canonical form used by this script."""
    key = key.lower().replace("@", "_")
    aliases = {
        "mpjpe_mm": "mpjpe",
        "pa_mpjpe_mm": "pa_mpjpe",
        "pck@50mm": "pck_50",
        "pck@100mm": "pck_100",
        "pck@150mm": "pck_150",
    }
    return aliases.get(key, key)


def _metric_value(raw: Any) -> Optional[float]:
    """Return a scalar float value, or None if not numeric."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_model_name(data: Dict[str, Any], path: Path) -> str:
    """Best-effort model/run name extraction from a result JSON."""
    if "model_config" in data and isinstance(data["model_config"], dict):
        model = data["model_config"].get("model")
        checkpoint = data["model_config"].get("checkpoint")
        if model:
            return str(model)
        if checkpoint:
            return Path(checkpoint).stem
    if "model" in data:
        return str(data["model"])
    return path.stem


def _extract_rows(data: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    """Normalize a result JSON into a flat list of table rows."""
    rows: List[Dict[str, Any]] = []
    model_name = _extract_model_name(data, path)

    # 1. WebBridge benchmark manifest: top-level "results" list.
    if "results" in data and isinstance(data["results"], list):
        for entry in data["results"]:
            if not isinstance(entry, dict):
                continue
            row: Dict[str, Any] = {
                "model": model_name,
                "dataset": entry.get("dataset", "-"),
                "sequence": entry.get("dataset", "-"),
            }
            for key, value in entry.items():
                if key in {"dataset", "path"}:
                    continue
                norm = _norm_metric_key(key)
                if _metric_value(value) is not None:
                    row[norm] = _metric_value(value)
            rows.append(row)
        return rows

    # 2. Auto-eval / dry-run benchmark manifest: nested datasets/sequences.
    if "datasets" in data and isinstance(data["datasets"], list):
        for dataset in data["datasets"]:
            if not isinstance(dataset, dict):
                continue
            ds_name = dataset.get("name", "-")
            # Use aggregated dataset metrics if sequences are absent.
            seqs = dataset.get("sequences", []) or []
            if not seqs and "metrics" in dataset:
                row = {"model": model_name, "dataset": ds_name, "sequence": "-"}
                for key, value in dataset.get("metrics", {}).items():
                    norm = _norm_metric_key(key)
                    if _metric_value(value) is not None:
                        row[norm] = _metric_value(value)
                rows.append(row)
                continue
            for seq in seqs:
                if not isinstance(seq, dict):
                    continue
                row = {
                    "model": model_name,
                    "dataset": ds_name,
                    "sequence": seq.get("name", seq.get("dataset", "-")),
                }
                for key, value in seq.get("metrics", {}).items():
                    norm = _norm_metric_key(key)
                    if _metric_value(value) is not None:
                        row[norm] = _metric_value(value)
                rows.append(row)
        return rows

    # 3. Generic per-condition metrics: keys are condition names, values are metric dicts.
    for key, value in data.items():
        if isinstance(value, dict):
            row = {"model": model_name, "dataset": key, "sequence": key}
            for mkey, mval in value.items():
                norm = _norm_metric_key(mkey)
                if _metric_value(mval) is not None:
                    row[norm] = _metric_value(mval)
            rows.append(row)

    return rows


def _collect_rows(inputs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level object in {path}")
        rows.extend(_extract_rows(data, path))
    return rows


def _select_metrics(rows: List[Dict[str, Any]], requested: Sequence[str]) -> List[str]:
    """Return the ordered list of metrics to display."""
    if requested:
        return [_norm_metric_key(m) for m in requested]
    available: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key in {"model", "dataset", "sequence"}:
                continue
            if key not in seen:
                seen.add(key)
                available.append(key)
    # Preserve preferred order for metrics that exist.
    ordered = [m for m in DEFAULT_METRICS if m in available]
    ordered += [m for m in available if m not in ordered]
    return ordered


def _format_cell(value: Any, metric: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        # PCK thresholds and AUC are unitless; MPJPEs are in mm.
        if metric.startswith("pck_") and metric != "pck_auc":
            return f"{value:.4f}"
        if metric == "pck_auc":
            return f"{value:.4f}"
        return f"{value:.2f}"
    return str(value)


def _build_markdown(rows: List[Dict[str, Any]], metrics: List[str]) -> str:
    headers = ["Model", "Dataset", "Sequence"] + [METRIC_LABELS.get(m, m) for m in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [
            str(row.get("model", "-")),
            str(row.get("dataset", "-")),
            str(row.get("sequence", "-")),
        ]
        for metric in metrics:
            cells.append(_format_cell(row.get(metric), metric))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_csv(rows: List[Dict[str, Any]], metrics: List[str]) -> List[List[str]]:
    headers = ["model", "dataset", "sequence"] + metrics
    result = [headers]
    for row in rows:
        result.append([str(row.get(h, "-")) for h in headers])
    return result


def _write_csv(path: Path, rows: List[Dict[str, Any]], metrics: List[str]) -> None:
    data = _build_csv(rows, metrics)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown/CSV benchmark table from result JSON files."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more benchmark result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown file. If omitted, the table is printed to stdout.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to include (default: auto-detect from JSON).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    rows = _collect_rows(args.inputs)
    if not rows:
        print("No rows extracted from inputs.", file=sys.stderr)
        sys.exit(1)

    metrics = _select_metrics(rows, args.metrics)
    table = _build_markdown(rows, metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"Wrote Markdown table to {args.output}")
    else:
        print(table)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.csv, rows, metrics)
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
