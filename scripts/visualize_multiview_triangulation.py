#!/usr/bin/env python3
"""Visualize multi-view 2D keypoints and triangulated 3D poses.

Loads a canonical ``.npz`` sample (``points_2d``, ``confidences``,
``joints_3d``, ``camera_K/R/t``), triangulates a 3D skeleton from the 2D
keypoints with confidence-weighted DLT, and renders:

1. A per-view 2D overlay of input keypoints, the triangulated skeleton
   reprojection, and the ground-truth skeleton reprojection.
2. A 3D skeleton plot (matplotlib) when matplotlib is available.

The 2D renderer uses only ``PIL`` so it works inside the project's virtual
environment even when matplotlib is linked against an incompatible NumPy build.

Example
-------
    python scripts/visualize_multiview_triangulation.py \\
        --sample data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \\
        --frame 0 \\
        --out_dir outputs/visualize_triangulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.triangulation import triangulate_dlt


# Human3.6M 17-joint layout used by the WebBridge/H36M samples.
JOINT_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "neck",
    "head",
    "head_top",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

# parent[i] is the parent joint of joint i; -1 marks the root.
PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

COLORS = {
    "input": (31, 119, 180),
    "triangulated": (214, 39, 40),
    "gt": (44, 160, 44),
}


def load_sample(path: str) -> dict:
    """Load a canonical multi-view ``.npz`` sample."""
    data = np.load(path, allow_pickle=True)
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Sample {path} missing keys: {missing}")
    return {
        "points_2d": data["points_2d"],
        "confidences": data["confidences"],
        "joints_3d": data["joints_3d"],
        "camera_K": data["camera_K"],
        "camera_R": data["camera_R"],
        "camera_t": data["camera_t"],
    }


def build_projection_matrices(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (V, 3, 4) projection matrices P = K [R | t]."""
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        Rt = np.hstack([R[v], t[v].reshape(3, 1)])
        P[v] = K[v] @ Rt
    return P


def project_points(joints_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project (J, 3) joints to (J, 2) image points."""
    X_cam = (R @ joints_3d.T).T + t
    z = X_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = (K @ np.hstack([X_cam[:, :2] / z, np.ones_like(z)]).T).T
    return uv[:, :2] / uv[:, 2:3]


def triangulate_frame(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    proj_matrices: np.ndarray,
) -> np.ndarray:
    """Triangulate a (J, 3) pose from per-view (V, J, 2) observations."""
    V, J, _ = points_2d.shape
    joints_3d = np.zeros((J, 3), dtype=np.float64)
    for j in range(J):
        if confidences[:, j].sum() <= 0:
            continue
        joints_3d[j] = triangulate_dlt(points_2d[:, j], proj_matrices, weights=confidences[:, j])
    return joints_3d


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    size: int = 5,
    fill: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
):
    x, y = xy
    draw.line([(x - size, y), (x + size, y)], fill=fill, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=fill, width=width)


def _draw_skeleton_pil(
    draw: ImageDraw.ImageDraw,
    points_2d: np.ndarray,
    color: tuple[int, int, int],
    point_radius: int = 4,
    line_width: int = 3,
):
    """Draw a 2D skeleton on a PIL ImageDraw context."""
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        draw.line(
            [(points_2d[parent, 0], points_2d[parent, 1]), (points_2d[child, 0], points_2d[child, 1])],
            fill=color,
            width=line_width,
        )
    for x, y in points_2d:
        r = point_radius
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline=color)


def _compute_canvas_transform(points: np.ndarray, width: int, height: int, margin: int = 40):
    """Compute a scale and offset that maps ``points`` onto a canvas."""
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    content_w = x_max - x_min
    content_h = y_max - y_min
    avail_w = width - 2 * margin
    avail_h = height - 2 * margin
    scale = min(avail_w / content_w, avail_h / content_h)
    offset_x = (width - scale * content_w) / 2 - scale * x_min
    offset_y = (height - scale * content_h) / 2 - scale * y_min
    return scale, offset_x, offset_y


def _apply_canvas_transform(
    points: np.ndarray,
    height: int,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    """Apply a previously computed canvas transform to a point set."""
    out = np.empty_like(points)
    out[:, 0] = points[:, 0] * scale + offset_x
    out[:, 1] = height - (points[:, 1] * scale + offset_y)
    return out


def _draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, canvas_width: int):
    """Draw a compact color legend in the top-right corner of a sub-image."""
    labels = [
        ("triang", COLORS["triangulated"]),
        ("input", COLORS["input"]),
        ("GT", COLORS["gt"]),
    ]
    x0 = canvas_width - 110
    y = 12
    box_w, box_h = 12, 12
    for text, color in labels:
        draw.rectangle([(x0, y), (x0 + box_w, y + box_h)], fill=color, outline=color)
        draw.text((x0 + box_w + 4, y - 1), text, fill=(0, 0, 0), font=font)
        y += box_h + 4


def render_per_view_overlay(
    points_2d: np.ndarray,
    triangulated_3d: np.ndarray,
    gt_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    confidences: np.ndarray,
    views: list[int] | None = None,
    canvas_size: tuple[int, int] = (400, 400),
) -> Image.Image:
    """Render a PIL grid of per-view 2D overlays with triangulated and GT skeletons."""
    V = K.shape[0]
    if views is None:
        views = list(range(V))

    reproj_tri = np.stack([project_points(triangulated_3d, K[v], R[v], t[v]) for v in views], axis=0)
    reproj_gt = np.stack([project_points(gt_3d, K[v], R[v], t[v]) for v in views], axis=0)

    n_views = len(views)
    n_cols = min(4, n_views)
    n_rows = (n_views + n_cols - 1) // n_cols
    w, h = canvas_size
    grid_w = n_cols * w
    grid_h = n_rows * (h + 30)
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
        title_font = font

    for idx, v in enumerate(views):
        # Compute a common framing from all points visible in this view.
        reference = np.concatenate([
            points_2d[v],
            reproj_tri[v],
            reproj_gt[v],
        ], axis=0)
        scale, offset_x, offset_y = _compute_canvas_transform(reference, w, h, margin=40)

        tri_px = _apply_canvas_transform(reproj_tri[v], h, scale, offset_x, offset_y)
        gt_px = _apply_canvas_transform(reproj_gt[v], h, scale, offset_x, offset_y)
        input_px = _apply_canvas_transform(points_2d[v], h, scale, offset_x, offset_y)

        sub = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(sub)

        # Ground-truth skeleton (behind everything).
        _draw_skeleton_pil(draw, gt_px, COLORS["gt"], point_radius=3, line_width=2)

        # Input keypoints and skeleton.
        _draw_skeleton_pil(draw, input_px, COLORS["input"], point_radius=3, line_width=2)

        # Triangulated reprojection on top.
        _draw_skeleton_pil(draw, tri_px, COLORS["triangulated"], point_radius=3, line_width=3)
        for x, y in tri_px:
            _draw_cross(draw, (x, y), size=5, fill=COLORS["triangulated"], width=2)

        # Title and compact legend.
        draw.text((10, 10), f"View {v}", fill=(0, 0, 0), font=title_font)
        _draw_legend(draw, font, w)

        col = idx % n_cols
        row = idx // n_cols
        grid.paste(sub, (col * w, row * (h + 30)))

    return grid


def plot_3d_skeleton(triangulated_3d: np.ndarray, gt_3d: np.ndarray, output_path: Path):
    """Render a 3D skeleton comparison using matplotlib if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib not available; skipping 3D skeleton plot.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    def _draw(joints, color):
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=40, alpha=0.7)
        for child, parent in enumerate(PARENTS):
            if parent < 0:
                continue
            xs = [joints[parent, 0], joints[child, 0]]
            ys = [joints[parent, 1], joints[child, 1]]
            zs = [joints[parent, 2], joints[child, 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.8)

    _draw(gt_3d, "green")
    _draw(triangulated_3d, "red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton: triangulated (red) vs ground truth (green)")
    ax.legend(["Ground truth", "Triangulated"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved 3D skeleton plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-view 2D keypoints and triangulated 3D poses."
    )
    parser.add_argument("--sample", required=True, help="Path to a canonical .npz sample.")
    parser.add_argument("--frame", type=int, default=0, help="Temporal frame index.")
    parser.add_argument("--views", type=int, nargs="+", default=None, help="View indices to render (default: all).")
    parser.add_argument("--out_dir", default="outputs/visualize_triangulation", help="Directory for output images.")
    parser.add_argument("--canvas_size", type=int, nargs=2, default=(400, 400), metavar=("W", "H"))
    args = parser.parse_args()

    sample = load_sample(args.sample)
    points_2d = sample["points_2d"]
    confidences = sample["confidences"]
    joints_3d = sample["joints_3d"]
    K = sample["camera_K"]
    R = sample["camera_R"]
    t = sample["camera_t"]

    if args.frame >= points_2d.shape[0]:
        raise ValueError(f"Frame {args.frame} out of range (0, {points_2d.shape[0] - 1})")

    proj_matrices = build_projection_matrices(K, R, t)
    tri_3d = triangulate_frame(points_2d[args.frame], confidences[args.frame], proj_matrices)
    gt_3d = joints_3d[args.frame]

    mpjpe = np.linalg.norm(tri_3d - gt_3d, axis=-1).mean()
    print(f"Frame {args.frame}: triangulation MPJPE vs GT = {mpjpe * 1000:.2f} mm (sample units assumed metres)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = render_per_view_overlay(
        points_2d=points_2d[args.frame],
        triangulated_3d=tri_3d,
        gt_3d=gt_3d,
        K=K,
        R=R,
        t=t,
        confidences=confidences[args.frame],
        views=args.views,
        canvas_size=tuple(args.canvas_size),
    )
    grid_path = out_dir / f"frame_{args.frame:05d}_2d_overlay.png"
    grid.save(grid_path)
    print(f"Saved per-view 2D overlay to {grid_path}")

    plot_3d_skeleton(tri_3d, gt_3d, out_dir / f"frame_{args.frame:05d}_3d_skeleton.png")


if __name__ == "__main__":
    main()
