"""Draw a clean architecture figure for the MotionFlow-MultiView paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, xy, w, h, text, color="white", edge="black", fontsize=9, bold=False):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def draw_arrow(ax, start, end, color="black"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Input
    draw_box(ax, (0.2, 3.8), 1.5, 1.0, "2D keypoints +\nconfidences\n(B, T, V, J, 3)", color="#E8F4FD")
    draw_box(ax, (0.2, 2.2), 1.5, 1.0, "Camera params\nK, R, t", color="#E8F4FD")

    # Encoder
    draw_box(ax, (2.2, 3.0), 1.6, 1.6, "Ray-aware\nencoder", color="#FFF3E0")

    # Attention stack
    draw_box(ax, (4.3, 3.0), 1.6, 1.6, "Joint attn +\nTemporal attn", color="#E8F5E9")

    # Weight head
    draw_box(ax, (6.4, 4.0), 1.6, 0.9, "Weight head\n+ conf", color="#FCE4EC")
    draw_box(ax, (6.4, 2.8), 1.6, 0.9, "Uncertainty\nsummary", color="#F3E5F5")

    # DLT
    draw_box(ax, (8.4, 3.5), 1.4, 1.2, "Weighted\nDLT", color="#E3F2FD")

    # Residual head
    draw_box(ax, (10.0, 3.0), 1.6, 1.6, "Residual\nMLP", color="#FFEBEE")

    # Output
    draw_box(ax, (11.2, 1.2), 0.6, 1.2, "X_ref\n(B, T, J, 3)", color="#E0F2F1")

    # Arrows
    draw_arrow(ax, (1.7, 4.3), (2.2, 4.0))
    draw_arrow(ax, (1.7, 2.7), (2.2, 3.4))
    draw_arrow(ax, (3.8, 3.8), (4.3, 3.8))
    draw_arrow(ax, (5.9, 3.8), (6.4, 4.4))
    draw_arrow(ax, (5.9, 3.4), (6.4, 3.2))
    draw_arrow(ax, (8.0, 4.1), (8.4, 4.0))
    draw_arrow(ax, (8.0, 3.2), (8.4, 3.8))
    draw_arrow(ax, (9.8, 4.1), (10.0, 3.8))
    draw_arrow(ax, (10.8, 3.0), (11.2, 1.8))

    ax.set_title("MotionFlow-MultiView: Temporal Ray-Attention Fusion with Residual Refinement",
                 fontsize=14, weight="bold", y=0.98)

    out_path = "docs/figures/architecture.png"
    import os
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved architecture figure to {out_path}")


if __name__ == "__main__":
    main()
