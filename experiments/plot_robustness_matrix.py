"""Plot a robustness matrix bar chart from a JSON metrics file.

Usage
-----
    python experiments/plot_robustness_matrix.py \
        --input outputs/eval_curriculum_robustness_s2_full.json \
        --output docs/figures/robustness_matrix.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot robustness matrix bar chart")
    parser.add_argument("--input", type=str, required=True, help="Robustness JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--title", type=str, default="Robustness Matrix")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    conditions = list(data.keys())
    mpjpe = [data[c]["mpjpe"] for c in conditions]
    pa_mpjpe = [data[c]["pa_mpjpe"] for c in conditions]

    x = range(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], mpjpe, width, label="MPJPE")
    ax.bar([i + width / 2 for i in x], pa_mpjpe, width, label="PA-MPJPE")

    ax.set_xlabel("Condition")
    ax.set_ylabel("Error (mm)")
    ax.set_title(args.title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved robustness plot to {out_path}")


if __name__ == "__main__":
    main()
