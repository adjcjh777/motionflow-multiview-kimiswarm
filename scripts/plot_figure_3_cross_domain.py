#!/usr/bin/env python3
"""Render Figure 3: cross-domain generalisation bar chart.

Reads evaluation JSONs for H36M true-GT test, MPI-INF-3DHP detected-2D,
AIST++-only -> H36M zero-shot transfer, and Shelf/Campus detected, then
produces a grouped bar chart comparing the DLT baseline with the v25, v85 and
v86 learned models.

Usage
-----
    python scripts/plot_figure_3_cross_domain.py
    python scripts/plot_figure_3_cross_domain.py --repo a800-D:/path/to/repo

Output
------
    docs/figures/cross_domain_generalisation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["H36M test", "MPI", "AIST++ -> H36M", "Shelf/Campus"]
METHODS = ["DLT", "v25 learned", "v85 learned", "v86 learned"]

# Default palette matched to the draft figures.
COLORS = {
    "DLT": "#4c78a8",
    "v25 learned": "#f58518",
    "v85 learned": "#54a24b",
    "v86 learned": "#e45756",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_h36m_test_mpjpe(repo: Path) -> Optional[float]:
    """TODO: replace with actual v25/v85/v86 H36M test JSON once available."""
    data = load_json(repo / "outputs" / "eval_v25_true_gt_v2_medium_a800_h36m_test.json")
    if data is None:
        return None
    # Flattened JSONs may use 'mpjpe_mm' or 'mpjpe'.
    return float(data.get("mpjpe_mm", data.get("mpjpe", np.nan)))


def get_mpi_mpjpe(repo: Path) -> Optional[float]:
    """DLT baseline mean MPJPE across the 16 MPI-INF-3DHP sequences."""
    data = load_json(
        repo / "outputs" / "mpi_rtmpose_detected_2d" / "dlt_baseline_detected_2d.json"
    )
    if data is None:
        return None
    return float(data.get("mean_mpjpe_mm", data.get("mean_mpjpe", np.nan)))


def get_aistpp_cross_mpjpe(repo: Path) -> Optional[float]:
    """AIST++-only model zero-shot evaluated on H36M S9/S11."""
    data = load_json(
        repo / "outputs" / "eval_aistpp_only_medium_a800_fast_v2_h36m_test.json"
    )
    if data is None:
        return None
    return float(data.get("combined_mpjpe_mm", data.get("mpjpe", np.nan)))


def get_shelf_campus_mpjpe(repo: Path) -> Optional[float]:
    """TODO: read Shelf/Campus eval JSON once available."""
    # Candidate output paths; update once the true-GT Shelf/Campus eval is run.
    candidates = [
        repo / "outputs" / "eval_v25_shelf_campus_detected.json",
        repo / "outputs" / "eval_v25_shelf_campus_true_gt.json",
    ]
    for path in candidates:
        data = load_json(path)
        if data is not None:
            return float(data.get("mpjpe_mm", data.get("mpjpe", np.nan)))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Render Figure 3 (cross-domain generalisation).")
    parser.add_argument("--repo", default=".", help="Repository root to read outputs from.")
    args = parser.parse_args()

    repo = Path(args.repo)

    # Raw values: rows=dataset, cols=method.
    values: Dict[str, Dict[str, Optional[float]]] = {
        "H36M test": {
            "DLT": 25.67,  # confidence-weighted DLT baseline on true-GT v2.
            "v25 learned": 30.83,  # v25 stability test (true-GT v1).
            "v85 learned": None,  # TODO: pending v85 test eval.
            "v86 learned": None,  # TODO: pending v86 test eval.
        },
        "MPI": {
            "DLT": get_mpi_mpjpe(repo),
            "v25 learned": None,  # TODO: run learned model on MPI detected-2D.
            "v85 learned": None,
            "v86 learned": None,
        },
        "AIST++ -> H36M": {
            "DLT": 15.93,  # AIST++ full-set confidence-weighted DLT baseline.
            "v25 learned": get_aistpp_cross_mpjpe(repo),
            "v85 learned": None,
            "v86 learned": None,
        },
        "Shelf/Campus": {
            "DLT": 132.29,  # confidence-weighted DLT on detected Shelf/Campus.
            "v25 learned": None,  # TODO.
            "v85 learned": None,
            "v86 learned": None,
        },
    }

    # Warn about TODOs.
    for dataset, method_values in values.items():
        for method, value in method_values.items():
            if value is None or np.isnan(value):
                print(f"[TODO] Missing data for {dataset} / {method}", file=sys.stderr)

    # Plot.
    x = np.arange(len(DATASETS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(METHODS):
        raw_heights = [values[dataset][method] for dataset in DATASETS]
        heights = [0.0 if h is None or np.isnan(h) else h for h in raw_heights]
        bars = ax.bar(x + i * width, heights, width, label=method, color=COLORS[method])
        for bar, raw in zip(bars, raw_heights):
            height = bar.get_height()
            if raw is None or np.isnan(raw):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 3,
                        "TODO", ha="center", va="bottom", fontsize=6, rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 3,
                        f"{height:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title("Cross-Domain Generalisation (lower is better)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = repo / "docs" / "figures" / "cross_domain_generalisation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 3 -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
