"""Plot MPJPE@k curves from variable-view evaluation results.

Usage:
    python experiments/plot_variable_views.py \
        --input results/variable_views.json --output figures/variable_views.png
    python experiments/plot_variable_views.py \
        --input results/variable_views.csv --output figures/variable_views.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        with open(path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    elif path.suffix.lower() == ".csv":
        results = {}
        with open(path) as f:
            header = next(f).strip().split(",")
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                k = int(parts[0])
                results[k] = {
                    "mean_mm": float(parts[1]) if len(parts) > 1 else 0.0,
                    "std_mm": float(parts[2]) if len(parts) > 2 else 0.0,
                    "n_subsets": int(parts[3]) if len(parts) > 3 else 0,
                }
        return results
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def plot_variable_views(results: dict, output_path: Path, title: str = "MPJPE@k"):
    k_values = sorted(results.keys())
    means = [results[k]["mean_mm"] for k in k_values]
    stds = [results[k].get("std_mm", 0.0) for k in k_values]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, means, marker="o", label="MPJPE@k")
    plt.fill_between(k_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)
    plt.xlabel("Number of active views (k)")
    plt.ylabel("MPJPE (mm)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .json or .csv results")
    parser.add_argument("--output", type=str, required=True, help="Output plot path")
    parser.add_argument("--title", type=str, default="MPJPE@k", help="Plot title")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    results = load_results(input_path)
    plot_variable_views(results, output_path, title=args.title)


if __name__ == "__main__":
    main()
