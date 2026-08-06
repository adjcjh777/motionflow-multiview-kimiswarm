"""Reusable figure generators for the MotionFlow-MultiView paper."""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SKELETON_JOINT_NAMES_17 = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "upper_neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def save(fig: plt.Figure, out_path: Path, *, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_main_mpjpe_bar(results: Dict[str, Dict], out_path: Path) -> None:
    """Bar chart of MPJPE (mm) per model variant."""
    names = [k.replace("_", "\n") for k in results.keys()]
    scores = [results[k]["mpjpe_mm"] for k in results.keys()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color="#1f77b4", edgecolor="black")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("MPI-INF-3DHP cross-subject MPJPE")
    ax.set_ylim(0, max(scores) * 1.15)
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save(fig, out_path)


def draw_per_joint_mpjpe(per_joint_errors: np.ndarray, out_path: Path, joint_names: List[str] = None) -> None:
    """Bar chart of per-joint MPJPE."""
    if joint_names is None:
        joint_names = [f"J{i}" for i in range(len(per_joint_errors))]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(per_joint_errors))
    ax.bar(x, per_joint_errors, color="#2ca02c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-joint MPJPE")
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_path)


def draw_pck_curve(thresholds: np.ndarray, pck_values: np.ndarray, out_path: Path, auc: float = None) -> None:
    """Plot PCK vs distance threshold."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, pck_values, "-o", color="#d62728", linewidth=2, markersize=4)
    ax.set_xlabel("Distance threshold (mm)")
    ax.set_ylabel("PCK")
    title = "PCK curve"
    if auc is not None:
        title += f" (AUC = {auc:.3f})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save(fig, out_path)


def draw_robustness_grid(robustness_report: Dict, out_path: Path) -> None:
    """Draw 1x3 robustness subplots from a standardized robustness_report dict."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    noise = robustness_report["noise"]
    axes[0].plot(
        [e["noise_std_px"] for e in noise],
        [e["mpjpe_mm"] for e in noise],
        "-o",
        color="#1f77b4",
    )
    axes[0].set_xlabel("Noise std (px)")
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("Gaussian noise")
    axes[0].grid(True, alpha=0.3)

    occ = robustness_report["occlusion"]
    axes[1].plot(
        [e["occlusion_rate"] * 100 for e in occ],
        [e["mpjpe_mm"] for e in occ],
        "-o",
        color="#2ca02c",
    )
    axes[1].set_xlabel("Occlusion rate (%)")
    axes[1].set_ylabel("MPJPE (mm)")
    axes[1].set_title("Random joint occlusion")
    axes[1].grid(True, alpha=0.3)

    out = robustness_report["outliers"]
    axes[2].plot(
        [e["outlier_rate"] * 100 for e in out],
        [e["mpjpe_mm"] for e in out],
        "-o",
        color="#d62728",
    )
    axes[2].set_xlabel("Outlier rate (%)")
    axes[2].set_ylabel("MPJPE (mm)")
    axes[2].set_title("2D outliers")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, out_path)


def draw_uncertainty_heatmap(weights: np.ndarray, out_path: Path, joint_names: List[str] = None) -> None:
    """Draw a per-view per-joint weight heatmap.

    Parameters
    ----------
    weights: (V, J) array of predicted weights.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xlabel("Joint")
    ax.set_ylabel("View")
    ax.set_title("Predicted per-view per-joint DLT weights")
    if joint_names is not None:
        ax.set_xticks(np.arange(len(joint_names)))
        ax.set_xticklabels(joint_names, rotation=45, ha="right")
    fig.colorbar(im, ax=ax)
    save(fig, out_path)
