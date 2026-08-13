#!/usr/bin/env python3
"""Render Figure 2: sparse-view MPJPE@k curves for true-GT H36M.

Reads the variable-view evaluation JSONs for v25, v85 and v86 and plots the
macro-averaged (S9/S11) MPJPE@k curves.  Missing or pending experiments are
logged as warnings and their curves are omitted.

Intended data flow
------------------
1. Training/eval produces JSONs under ``outputs/`` and
   ``outputs/variable_view_fix/``.
2. ``scripts/analyze_v86_ablation.py`` summarises these JSONs.
3. This script reads the same JSONs and renders the figure.

Usage
-----
    python scripts/plot_figure_2_sparse_view.py
    python scripts/plot_figure_2_sparse_view.py --repo a800-D:/path/to/repo

Output
------
    docs/figures/h36m_true_gt_mpjpe_at_k.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


K_VALUES = [2, 3, 4]

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_per_dataset(path: Path) -> Optional[Dict[str, Dict[int, float]]]:
    """Load a variable-view JSON and return {dataset: {k: mpjpe_at_k}}."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    per_dataset = data.get("per_dataset", {})
    out: Dict[str, Dict[int, float]] = {}
    for dataset, k_map in per_dataset.items():
        out[dataset] = {}
        for k_str, metrics in k_map.items():
            try:
                k = int(k_str)
            except ValueError:
                continue
            mpjpe = metrics.get("mpjpe_at_k", metrics.get("mean_mm"))
            if mpjpe is None:
                continue
            out[dataset][k] = float(mpjpe)
    return out


def macro_mean(path: Path) -> Dict[int, float]:
    """Return k -> mean MPJPE over S9 and S11, or NaN if missing."""
    data = load_per_dataset(path)
    if data is None:
        return {k: np.nan for k in K_VALUES}
    out = {}
    for k in K_VALUES:
        vals = [data[subject][k] for subject in ("S9", "S11")
                if subject in data and k in data[subject]]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def plot_series(ax, values: Dict[int, float], label: str, style: str,
                color: str, marker: str = "o", linewidth: float = 2.0):
    ks = [k for k in K_VALUES if not np.isnan(values[k])]
    ys = [values[k] for k in ks]
    if ks:
        ax.plot(ks, ys, style, marker=marker, label=label, color=color,
                linewidth=linewidth, markersize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Render Figure 2 (sparse-view MPJPE@k).")
    parser.add_argument("--repo", default=".", help="Repository root to read outputs from.")
    args = parser.parse_args()

    repo = Path(args.repo)
    vv_fix = repo / "outputs" / "variable_view_fix"
    vv_out = repo / "outputs"

    # Model-agnostic baselines (frozen values from docs/results_true_gt_h36m.md).
    dlt_conf = {2: 36.42, 3: 33.68, 4: 25.94}
    iskakov = {2: 53.62, 3: 27.84, 4: 23.42}

    # Learned / hybrid curves.
    learned = {
        "v25 stability (DLT fallback k<4)":
            (vv_fix / "variable_view_v25_true_gt_stability_a800_dlt_fallback.json", "-", "#f58518", "o"),
        "v85 random dropout (no fallback)":
            (vv_out / "variable_view_v85_random_view_dropout_medium_a800.json", "-", "#54a24b", "s"),
        "v85 random dropout (DLT fallback)":
            (vv_fix / "variable_view_v85_random_view_dropout_medium_a800_dlt_fallback.json", "--", "#2d8a4e", "s"),
        "v86 no count embedding (no fallback)":
            (vv_out / "variable_view_v86_no_count_embedding_medium_a800.json", "-", "#e45756", "d"),
        "v86 no count embedding (DLT fallback)":
            (vv_fix / "variable_view_v86_no_count_embedding_medium_a800_dlt_fallback.json", "--", "#b33030", "d"),
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Baselines.
    plot_series(ax, dlt_conf, "DLT (confidence-weighted)", "-", "#2d5f8f", marker="s")
    plot_series(ax, iskakov, "Iskakov ICCV 2019", "-", "#9b59b6", marker="^")

    # Learned curves.
    for label, (path, style, color, marker) in learned.items():
        values = macro_mean(path)
        if all(np.isnan(v) for v in values.values()):
            print(f"[TODO] Missing data for '{label}': {path}", file=sys.stderr)
            continue
        plot_series(ax, values, label, style, color, marker=marker)

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Number of active views (k)", fontsize=12)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title("True-GT H36M Sparse-View Robustness (S9/S11 macro mean)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = repo / "docs" / "figures" / "h36m_true_gt_mpjpe_at_k.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 2 -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
