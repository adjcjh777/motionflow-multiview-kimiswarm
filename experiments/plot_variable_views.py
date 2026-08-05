"""Plot a variable-view MPJPE@k curve from eval_variable_views JSON output.

Usage:
    python experiments/plot_variable_views.py \
        --json outputs/variable_views_crossview_residual_smoke.json \
        --out docs/figures/variable_view_crossview_residual.png
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.json) as f:
        results = json.load(f)

    ks = sorted(int(k) for k in results.keys())
    means = [results[str(k)]["mean_mm"] for k in ks]
    stds = [results[str(k)]["std_mm"] for k in ks]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.errorbar(ks, means, yerr=stds, marker="o", capsize=5)
    plt.xlabel("Number of active views")
    plt.ylabel("MPJPE (mm)")
    plt.title("Variable-view inference performance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
