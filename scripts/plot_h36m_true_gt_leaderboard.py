#!/usr/bin/env python
"""Render the H36M true-GT leaderboard as a matplotlib bar chart.

Reads `docs/results_true_gt_h36m.md`, extracts the "Current results" table,
and writes a horizontal bar chart of combined direct MPJPE.

Usage:
    python scripts/plot_h36m_true_gt_leaderboard.py
    python scripts/plot_h36m_true_gt_leaderboard.py \
        --input docs/results_true_gt_h36m.md \
        --output docs/figures/h36m_true_gt_leaderboard.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def find_current_results_table(text: str) -> tuple:
    """Locate the 'Current results' markdown table and return (headers, rows).

    Returns
    -------
    headers : list[str]
    rows : list[list[str]]
    """
    # Find the section heading
    match = re.search(r"## Current results\n\n(.*?)(?:\n## |\Z)", text, re.DOTALL)
    if not match:
        raise ValueError("Could not find '## Current results' section")

    section = match.group(1)
    lines = [line for line in section.splitlines() if line.strip()]


    # First two lines should be header and separator
    if len(lines) < 3:
        raise ValueError("Current results table is too short")

    headers = [c.strip() for c in lines[0].split("|") if c.strip()]
    rows = []
    for line in lines[2:]:
        # Stop at the next section or empty line
        if line.startswith("##") or not line.strip():
            break
        cells = [c.strip() for c in line.split("|")][1:-1]
        if len(cells) != len(headers):
            continue
        rows.append(cells)

    return headers, rows


def parse_leaderboard(rows: list, headers: list) -> list[dict]:
    """Convert raw markdown rows into a list of structured method dicts."""
    methods = []
    # Map likely column names robustly
    col_map = {
        "method": ["Method"],
        "s9": ["S9 direct (mm)", "S9"],
        "s11": ["S11 direct (mm)", "S11"],
        "combined": ["Combined direct (mm)", "Combined"],
        "pa": ["Combined PA-MPJPE (mm)", "PA-MPJPE"],
    }

    def find_col(key_candidates):
        for candidate in key_candidates:
            try:
                return headers.index(candidate)
            except ValueError:
                continue
        raise ValueError(f"Missing column matching {key_candidates}")

    method_idx = find_col(col_map["method"])
    combined_idx = find_col(col_map["combined"])
    s9_idx = find_col(col_map["s9"])
    s11_idx = find_col(col_map["s11"])

    for row in rows:
        method = row[method_idx].strip()
        if not method:
            continue
        try:
            combined = float(row[combined_idx])
            s9 = float(row[s9_idx])
            s11 = float(row[s11_idx])
        except (ValueError, IndexError):
            continue
        methods.append({
            "method": method,
            "combined": combined,
            "s9": s9,
            "s11": s11,
        })

    return methods


def classify_method(method: str) -> str:
    """Return a high-level category for coloring."""
    lower = method.lower()
    if any(x in lower for x in ["dlc", "dlc", "ransac"]):
        return "geometric"
    if "iskakov" in lower:
        return "geometric"
    return "learned"


def plot_leaderboard(methods: list, output_path: Path) -> None:
    """Render a horizontal bar chart sorted by combined MPJPE."""
    methods = sorted(methods, key=lambda m: m["combined"], reverse=True)

    labels = [m["method"] for m in methods]
    values = [m["combined"] for m in methods]
    colors = ["#4c78a8" if classify_method(m["method"]) == "geometric" else "#f58518" for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(6, len(methods) * 0.45)))
    bars = ax.barh(labels, values, color=colors)

    # Annotate bars with combined MPJPE value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Combined direct MPJPE (mm)", fontsize=11)
    ax.set_title("H36M True-GT Standard Protocol Leaderboard", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.15 if values else 1)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4c78a8", label="Geometric / triangulation baseline"),
        Patch(facecolor="#f58518", label="Learned MotionFlow variant"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved leaderboard chart -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render H36M true-GT leaderboard bar chart from markdown results."
    )
    parser.add_argument(
        "--input",
        default="docs/results_true_gt_h36m.md",
        help="Path to results_true_gt_h36m.md",
    )
    parser.add_argument(
        "--output",
        default="docs/figures/h36m_true_gt_leaderboard.png",
        help="Output image path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    headers, rows = find_current_results_table(text)
    methods = parse_leaderboard(rows, headers)

    if not methods:
        print("No methods parsed from the leaderboard table.")
        return

    plot_leaderboard(methods, output_path)


if __name__ == "__main__":
    main()
