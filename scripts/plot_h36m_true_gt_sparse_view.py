#!/usr/bin/env python
"""Render the H36M true-GT sparse-view MPJPE@k comparison figure.

Reads the per-subject variable-view JSONs in
``outputs/variable_view_fix/`` and plots macro-averaged MPJPE@k for the
direct DLT baseline, Iskakov ICCV 2019, and the MotionFlow variants that
fall back to confidence-weighted DLT for k<4.

Usage:
    python scripts/plot_h36m_true_gt_sparse_view.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


K_VALUES = [2, 3, 4]


def macro_from_per_dataset(path: Path) -> dict[int, float]:
    """Average per-subject MPJPE@k across S9 and S11."""
    with open(path) as f:
        data = json.load(f)
    per_dataset = data.get("per_dataset", {})
    out: dict[int, float] = {}
    for k in K_VALUES:
        vals = [
            per_dataset[subject][str(k)]["mpjpe_at_k"]
            for subject in ("S9", "S11")
            if subject in per_dataset and str(k) in per_dataset[subject]
        ]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def plot_series(ax, values: dict[int, float], label: str, style: str, color: str, marker: str = "o"):
    ks = [k for k in K_VALUES if not np.isnan(values[k])]
    ys = [values[k] for k in ks]
    ax.plot(ks, ys, style, marker=marker, label=label, color=color, linewidth=2, markersize=7)


def main() -> None:
    base = Path("outputs/variable_view_fix")

    # Model-agnostic confidence-weighted DLT fallback values (model-independent,
    # used as the k<4 fallback for v25/v81/v82; source: docs/results_true_gt_h36m.md).
    dlt_conf = {2: 36.42, 3: 33.68, 4: 25.94}
    # Unweighted DLT baseline.
    dlt_unweighted = {2: 37.19, 3: 34.86, 4: 29.15}
    # Iskakov ICCV 2019 learnable triangulation (source: docs/results_true_gt_h36m.md).
    iskakov = {2: 53.62, 3: 27.84, 4: 23.42}

    v25 = macro_from_per_dataset(base / "variable_view_v25_true_gt_stability_a800_dlt_fallback.json")
    v82 = macro_from_per_dataset(base / "variable_view_v82_true_gt_medium_a800_dlt_fallback.json")

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_series(ax, dlt_unweighted, "DLT (unweighted)", "--", "#4c78a8", marker="s")
    plot_series(ax, dlt_conf, "DLT (confidence-weighted)", "-", "#2d5f8f", marker="s")
    plot_series(ax, iskakov, "Iskakov ICCV 2019", "-", "#54a24b", marker="^")
    plot_series(ax, v25, "v25 stability (DLT fallback k<4)", "-", "#f58518", marker="o")
    plot_series(ax, v82, "v82 multi-scale temporal (DLT fallback k<4)", "-", "#e45756", marker="d")

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Number of active views (k)", fontsize=11)
    ax.set_ylabel("MPJPE (mm)", fontsize=11)
    ax.set_title("True-GT H36M Sparse-View Robustness (S9/S11 macro mean)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(max(dlt_unweighted.values()), max(iskakov.values()), max(v25.values())) * 1.1)

    plt.tight_layout()
    output_path = Path("docs/figures/h36m_true_gt_mpjpe_at_k.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved sparse-view plot -> {output_path}")


if __name__ == "__main__":
    main()
