"""Draw a bar chart comparing MPI-INF-3DHP MPJPE for the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    models = [
        "DLT baseline\n(25.21 mm)",
        "Temporal ray-attn\n(25.21 mm)",
        "Residual small\n(13.22 mm)",
        "Residual full\n(11.17 mm)",
    ]
    mpjpe = [25.21, 25.21, 13.22, 11.17]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, mpjpe, color=colors, edgecolor="black")
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title("MPI-INF-3DHP cross-subject MPJPE", fontsize=14, weight="bold")
    ax.set_ylim(0, 30)
    for bar, val in zip(bars, mpjpe):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.axhline(11.17, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    out = "docs/figures/mpi_mpjpe_bar.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
