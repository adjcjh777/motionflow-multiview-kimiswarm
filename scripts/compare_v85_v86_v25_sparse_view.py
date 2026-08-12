#!/usr/bin/env python3
"""Compare sparse-view (k=2,3,4) results across v85, v86 and v25 baselines.

Reads JSON/CSV variable-view evaluation outputs for each baseline, extracts
MPJPE@k for H36M subjects S9 and S11, and emits:

* A comparison table to stdout.
* A bar plot at ``outputs/v85_v86_v25_sparse_view_comparison.png``.
* A JSON report at ``outputs/v85_v86_v25_sparse_view_comparison.json``.
* A Markdown report at ``docs/v85_v86_v25_sparse_view_comparison.md``.

Usage
-----
    python scripts/compare_v85_v86_v25_sparse_view.py \
        --v85-no-fallback outputs/variable_view_v85_random_view_dropout_medium_a800.json \
        --v85-dlt-fallback outputs/variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json \
        --v86 outputs/variable_view_v86_no_count_embedding_medium_a800_dlt_fallback.json \
        --v25-dlt-fallback outputs/variable_view_fix/variable_view_v25_true_gt_stability_a800_dlt_fallback.json

Any input may be omitted; missing baselines are reported as ``N/A`` and skipped
in the plot.  CSV inputs are supported provided they contain at least the
columns ``dataset,k,mpjpe_at_k`` (or ``dataset``, ``k``, ``mpjpe_mm``,
``mean_mm``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional matplotlib with a helpful error message.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for compare_v85_v86_v25_sparse_view.py. "
        "Install it with: pip install matplotlib"
    ) from exc


DEFAULT_OUT_DIR = Path("outputs")
DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_PLOT_NAME = "v85_v86_v25_sparse_view_comparison.png"
DEFAULT_JSON_NAME = "v85_v86_v25_sparse_view_comparison.json"
DEFAULT_MD_NAME = "v85_v86_v25_sparse_view_comparison.md"

DATASETS = ["S9", "S11"]
K_VALUES = [2, 3, 4]


@dataclass
class SparseViewResult:
    """Holds MPJPE@k for a single baseline."""

    name: str
    s9: Dict[int, Optional[float]] = field(default_factory=dict)
    s11: Dict[int, Optional[float]] = field(default_factory=dict)
    missing: bool = True
    note: str = ""

    def get(self, dataset: str, k: int) -> Optional[float]:
        store = self.s9 if dataset == "S9" else self.s11
        return store.get(k)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _extract_json(path: Path) -> Dict[str, Dict[int, float]]:
    """Parse the standard per_dataset JSON format.

    Returns a nested dict ``{dataset: {k: mpjpe_at_k}}``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, float]] = {}
    per_dataset = data.get("per_dataset", {})
    for dataset in DATASETS:
        out[dataset] = {}
        if dataset not in per_dataset:
            continue
        for k in K_VALUES:
            entry = per_dataset[dataset].get(str(k)) or per_dataset[dataset].get(k)
            if entry is None:
                continue
            # Accept either ``mpjpe_at_k`` or ``mean_mm``/``mpjpe_mm``.
            for key in ("mpjpe_at_k", "mean_mm", "mpjpe_mm", "mpjpe"):
                if key in entry:
                    out[dataset][k] = float(entry[key])
                    break
    return out


def _extract_csv(path: Path) -> Dict[str, Dict[int, float]]:
    """Parse a CSV file with columns containing dataset, k and mpjpe_at_k."""
    out: Dict[str, Dict[int, float]] = {d: {} for d in DATASETS}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        fields_lower = [f.lower() for f in reader.fieldnames]

        # Try to identify the relevant columns heuristically.
        def _col(*candidates: str) -> Optional[str]:
            for cand in candidates:
                if cand in reader.fieldnames:
                    return cand
                lc = cand.lower()
                for orig, low in zip(reader.fieldnames, fields_lower):
                    if low == lc:
                        return orig
            return None

        k_col = _col("k", "n_views", "num_views")
        dataset_col = _col("dataset", "subject", "seq")
        mpjpe_col = _col("mpjpe_at_k", "mpjpe_mm", "mean_mm", "mpjpe")

        if not all([k_col, dataset_col, mpjpe_col]):
            raise ValueError(
                f"CSV {path} is missing one of required columns. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            dataset = row[dataset_col].strip()
            if dataset not in DATASETS:
                continue
            try:
                k = int(row[k_col])
                mpjpe = float(row[mpjpe_col])
            except (ValueError, KeyError):
                continue
            if k in K_VALUES:
                out[dataset][k] = mpjpe
    return out


def _extract(path: Optional[Path]) -> Tuple[bool, Optional[Dict[str, Dict[int, float]]], str]:
    """Return (success, data, note) for a single input path."""
    if path is None:
        return False, None, "not provided"
    if not path.exists():
        return False, None, f"file not found: {path}"
    try:
        if path.suffix.lower() == ".csv":
            data = _extract_csv(path)
        else:
            data = _extract_json(path)
        return True, data, f"loaded {path}"
    except Exception as exc:  # pragma: no cover - safety net
        return False, None, f"failed to parse {path}: {exc}"


def _build_result(
    name: str, path: Optional[Path]
) -> SparseViewResult:
    result = SparseViewResult(name=name)
    success, data, note = _extract(path)
    result.note = note
    if not success or data is None:
        return result
    result.missing = False
    for dataset in DATASETS:
        store = result.s9 if dataset == "S9" else result.s11
        store.update(data.get(dataset, {}))
    return result


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _print_table(results: List[SparseViewResult]) -> str:
    """Render and return a Markdown/stdout table."""
    lines: List[str] = []
    header = ["k", "Subject"] + [r.name for r in results]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for k in K_VALUES:
        for subject in ["S9", "S11"]:
            row = [str(k), subject]
            for r in results:
                row.append(_format_value(r.get(subject, k)))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _plot(results: List[SparseViewResult], out_path: Path) -> None:
    """Create a grouped bar plot comparing sparse-view MPJPE."""
    available = [r for r in results if not r.missing]
    if not available:
        _warn("No results available to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Number of bars per k: one per baseline.
    x = np.arange(len(K_VALUES))
    width = 0.8 / max(len(available), 1)
    colors = plt.get_cmap("tab10")

    for ax_idx, subject in enumerate(["S9", "S11"]):
        ax = axes[ax_idx]
        for i, result in enumerate(available):
            values = [result.get(subject, k) for k in K_VALUES]
            values = [v if v is not None else np.nan for v in values]
            offset = (i - len(available) / 2) * width + width / 2
            ax.bar(
                x + offset,
                values,
                width,
                label=result.name,
                color=colors(i),
            )

        ax.set_xlabel("Active views (k)")
        ax.set_ylabel("MPJPE (mm)")
        ax.set_title(f"Sparse-view MPJPE — {subject}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 1:
            ax.legend(loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _json_report(results: List[SparseViewResult]) -> Dict[str, object]:
    report: Dict[str, object] = {}
    for r in results:
        entry: Dict[str, object] = {
            "missing": r.missing,
            "note": r.note,
        }
        for dataset in DATASETS:
            store = r.s9 if dataset == "S9" else r.s11
            entry[dataset] = {str(k): store.get(k) for k in K_VALUES}
        report[r.name] = entry
    return report


def _markdown_report(
    results: List[SparseViewResult],
    generated_time: str,
    plot_path: Path,
    json_path: Path,
) -> str:
    lines: List[str] = [
        "# Sparse-View Comparison: v85, v86 and v25 DLT-Fallback",
        "",
        f"Generated: {generated_time}",
        "",
        "## Results (MPJPE @ k)",
        "",
    ]
    lines.append(_print_table(results))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for r in results:
        status = "missing" if r.missing else "loaded"
        lines.append(f"* **{r.name}**: {status} — {r.note}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"* Plot: `{plot_path}`")
    lines.append(f"* JSON report: `{json_path}`")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sparse-view (k=2,3,4) results across v85, v86 and v25."
    )
    parser.add_argument(
        "--v85-no-fallback",
        type=Path,
        default=None,
        help="Path to v85 no-fallback variable-view JSON/CSV.",
    )
    parser.add_argument(
        "--v85-dlt-fallback",
        type=Path,
        default=None,
        help="Path to v85 DLT-fallback variable-view JSON/CSV.",
    )
    parser.add_argument(
        "--v86",
        type=Path,
        default=None,
        help="Path to v86 variable-view JSON/CSV (optional).",
    )
    parser.add_argument(
        "--v25-dlt-fallback",
        type=Path,
        default=None,
        help="Path to v25 DLT-fallback baseline JSON/CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for PNG and JSON outputs (default: outputs).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=f"Override plot output path (default: {DEFAULT_OUT_DIR / DEFAULT_PLOT_NAME}).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help=f"Override JSON report path (default: {DEFAULT_OUT_DIR / DEFAULT_JSON_NAME}).",
    )
    parser.add_argument(
        "--md-path",
        type=Path,
        default=None,
        help=f"Override Markdown report path (default: {DEFAULT_DOCS_DIR / DEFAULT_MD_NAME}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the comparison table to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    plot_path = args.plot_path or args.out_dir / DEFAULT_PLOT_NAME
    json_path = args.json_path or args.out_dir / DEFAULT_JSON_NAME
    md_path = args.md_path or DEFAULT_DOCS_DIR / DEFAULT_MD_NAME

    results = [
        _build_result("v85 no-fallback", args.v85_no_fallback),
        _build_result("v85 DLT-fallback", args.v85_dlt_fallback),
        _build_result("v86", args.v86),
        _build_result("v25 DLT-fallback", args.v25_dlt_fallback),
    ]

    for r in results:
        if r.missing:
            _warn(f"{r.name}: {r.note}")

    # stdout table
    table = _print_table(results)
    if not args.quiet:
        print(table)

    # Plot
    try:
        _plot(results, plot_path)
    except Exception as exc:  # pragma: no cover
        _warn(f"Could not create plot: {exc}")

    # JSON report
    report = _json_report(results)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown report
    from datetime import datetime, timezone

    md_text = _markdown_report(
        results,
        generated_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        plot_path=plot_path,
        json_path=json_path,
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
