"""Draw robustness charts for the paper."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    report_path = Path("outputs/robustness_residual_final5/robustness_report.json")
    if not report_path.exists():
        print(f"{report_path} not found")
        return
    with open(report_path) as f:
        report = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Noise
    noise = report["noise"]
    x = [e["noise_std_px"] for e in noise]
    y = [e["mpjpe_mm"] for e in noise]
    axes[0].plot(x, y, "-o", color="#1f77b4", linewidth=2, markersize=6)
    axes[0].set_xlabel("Gaussian noise std (px)")
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("Gaussian noise")
    axes[0].grid(True, alpha=0.3)

    # Occlusion
    occ = report["occlusion"]
    x = [e["occlusion_rate"] * 100 for e in occ]
    y = [e["mpjpe_mm"] for e in occ]
    axes[1].plot(x, y, "-o", color="#2ca02c", linewidth=2, markersize=6)
    axes[1].set_xlabel("Occlusion rate (%)")
    axes[1].set_ylabel("MPJPE (mm)")
    axes[1].set_title("Random joint occlusion")
    axes[1].grid(True, alpha=0.3)

    # Outliers
    out = report["outliers"]
    x = [e["outlier_rate"] * 100 for e in out]
    y = [e["mpjpe_mm"] for e in out]
    axes[2].plot(x, y, "-o", color="#d62728", linewidth=2, markersize=6)
    axes[2].set_xlabel("Outlier rate (%)")
    axes[2].set_ylabel("MPJPE (mm)")
    axes[2].set_title("2D outliers")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Robustness of RayAttentionTemporalResidual (final5)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "docs/figures/robustness_final5.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
