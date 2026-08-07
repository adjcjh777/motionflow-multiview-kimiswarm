#!/usr/bin/env python3
"""Render multi-view skeleton projections for a WebBridge sample.

The script loads a canonical ``.npz`` sample (``points_2d``, ``joints_3d``,
``camera_K/R/t``), projects a 3D pose into every requested camera view, and
saves a multi-view grid image with the projected skeleton overlaid.

The renderer uses only ``PIL`` and ``numpy``, so it works inside the
project's virtual environment even when the global matplotlib is linked
against an incompatible NumPy build.

Example
-------
    python scripts/visualize_multiview_pose.py \
        --sample data/webbridge/h36m_meters/s_01_acts_10_multiview_m.npz \
        --frame 0 --views 0 1 2 3 \
        --out_dir outputs/visualizations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Canonical H36M 17-joint layout (matches WebBridge mixed loader).
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
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "head_top",
]

# Parent indices for the H36M 17-joint skeleton (-1 for root).
PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 15]

# Distinct colors for left/right limbs.
DEFAULT_LIMB_COLOR = "#1f77b4"
LEFT_COLOR = "#d62728"
RIGHT_COLOR = "#2ca02c"


# ---------------------------------------------------------------------------
# Projection & drawing helpers
# ---------------------------------------------------------------------------


def _limb_color(j: int) -> str:
    """Color a limb by its child joint: left=red, right=green, otherwise blue."""
    name = JOINT_NAMES[j]
    if "left" in name:
        return LEFT_COLOR
    if "right" in name:
        return RIGHT_COLOR
    return DEFAULT_LIMB_COLOR


def project_pose(
    joints_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project ``(J, 3)`` 3D joints into ``(J, 2)`` image points.

    Parameters
    ----------
    joints_3d: (J, 3)
    K: (3, 3) intrinsics
    R: (3, 3) rotation
    t: (3,) translation

    Returns
    -------
    (J, 2) projected 2D points.
    """
    Xc = (R @ joints_3d.T).T + t  # camera-space points
    z = Xc[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = (K @ np.hstack([Xc[:, :2] / z, np.ones_like(z)]).T).T
    return uv[:, :2] / uv[:, 2:3]


def _fit_to_canvas(
    points: np.ndarray,
    canvas_size: tuple[int, int],
    margin: int = 30,
) -> np.ndarray:
    """Map an arbitrary 2D point cloud to pixel coordinates on the canvas."""
    h, w = canvas_size
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0

    # Keep aspect ratio; fit within margin.
    content_w = x_max - x_min
    content_h = y_max - y_min
    avail_w = w - 2 * margin
    avail_h = h - 2 * margin
    scale = min(avail_w / content_w, avail_h / content_h)

    offset_x = (w - scale * content_w) / 2 - scale * x_min
    offset_y = (h - scale * content_h) / 2 - scale * y_min

    uv = np.empty_like(points)
    uv[:, 0] = points[:, 0] * scale + offset_x
    # Invert y so that image coordinates grow downward.
    uv[:, 1] = h - (points[:, 1] * scale + offset_y)
    return uv


def _draw_cross(draw: ImageDraw.ImageDraw, xy: tuple[float, float], size: int = 5, fill: str = "#000000", width: int = 2):
    x, y = xy
    draw.line([(x - size, y), (x + size, y)], fill=fill, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=fill, width=width)


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    points_2d: np.ndarray,
    color: str = DEFAULT_LIMB_COLOR,
    point_radius: int = 4,
    line_width: int = 3,
):
    """Draw a 2D skeleton on an existing ImageDraw context."""
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        pts = np.stack([points_2d[parent], points_2d[child]], axis=0)
        draw.line(
            [(pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1])],
            fill=_limb_color(child),
            width=line_width,
        )

    for x, y in points_2d:
        r = point_radius
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline=color)


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------


def render_multiview_grid(
    sample: dict,
    frame: int = 0,
    pose3d_pred: np.ndarray | None = None,
    views: list[int] | None = None,
    show_gt_keypoints: bool = True,
    canvas_size: tuple[int, int] = (400, 400),
) -> Image.Image:
    """Create a multi-view grid image for a WebBridge sample.

    Parameters
    ----------
    sample: dict loaded from a canonical .npz file.
    frame: temporal frame index.
    pose3d_pred: optional (J, 3) predicted pose. If None, uses ``joints_3d``.
    views: list of view indices to render. Defaults to all available views.
    show_gt_keypoints: overlay ground-truth 2D keypoints if available.
    canvas_size: (width, height) for each sub-image.

    Returns
    -------
    PIL Image.
    """
    points_2d = sample["points_2d"]
    joints_3d = sample["joints_3d"]
    K = sample["camera_K"]
    R = sample["camera_R"]
    t = sample["camera_t"]

    n_views_total = K.shape[0]
    views = list(range(n_views_total)) if views is None else views

    if pose3d_pred is None:
        pose3d_pred = joints_3d[frame]
    else:
        if pose3d_pred.ndim == 3:
            pose3d_pred = pose3d_pred[frame]

    gt_3d = joints_3d[frame]

    n_cols = min(4, len(views))
    n_rows = (len(views) + n_cols - 1) // n_cols
    w, h = canvas_size
    grid_w = n_cols * w
    grid_h = n_rows * h + n_rows * 30  # extra row height for titles
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
        title_font = font

    for idx, view_idx in enumerate(views):
        pred_uv = project_pose(pose3d_pred, K[view_idx], R[view_idx], t[view_idx])
        gt_uv = project_pose(gt_3d, K[view_idx], R[view_idx], t[view_idx])

        # Reference points for framing.
        reference = np.concatenate([gt_uv, pred_uv], axis=0) if show_gt_keypoints else pred_uv

        sub = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(sub)

        # Compute a common scale/offset so both GT and predicted joints fit.
        x_min, x_max = reference[:, 0].min(), reference[:, 0].max()
        y_min, y_max = reference[:, 1].min(), reference[:, 1].max()
        if x_min == x_max:
            x_min, x_max = x_min - 1.0, x_max + 1.0
        if y_min == y_max:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        content_w = x_max - x_min
        content_h = y_max - y_min
        avail_w = w - 2 * 40
        avail_h = h - 2 * 40
        scale = min(avail_w / content_w, avail_h / content_h)
        offset_x = (w - scale * content_w) / 2 - scale * x_min
        offset_y = (h - scale * content_h) / 2 - scale * y_min

        def _map(pts: np.ndarray) -> np.ndarray:
            out = np.empty_like(pts)
            out[:, 0] = pts[:, 0] * scale + offset_x
            out[:, 1] = h - (pts[:, 1] * scale + offset_y)
            return out

        pred_px = _map(pred_uv)
        gt_px = _map(gt_uv) if show_gt_keypoints else None

        if show_gt_keypoints:
            for x, y in gt_px:
                _draw_cross(draw, (x, y), size=5, fill="#aaaaaa", width=2)

        _draw_skeleton(draw, pred_px, color="#ff7f0e")

        # Title bar at top.
        title = f"View {view_idx}"
        draw.text((10, 10), title, fill="#000000", font=title_font)
        draw.text((w - 120, 10), "Predicted", fill="#ff7f0e", font=font)

        col = idx % n_cols
        row = idx // n_cols
        grid.paste(sub, (col * w, row * (h + 30)))

    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render multi-view projected skeletons for a sample."
    )
    parser.add_argument("--sample", required=True, help="Path to a canonical .npz sample.")
    parser.add_argument("--frame", type=int, default=0, help="Temporal frame index.")
    parser.add_argument(
        "--pose3d",
        default=None,
        help="Optional .npy file with predicted 3D pose (T, J, 3) or (J, 3).",
    )
    parser.add_argument(
        "--views",
        type=int,
        nargs="+",
        default=None,
        help="View indices to render (default: all).",
    )
    parser.add_argument(
        "--no-gt-keypoints",
        action="store_true",
        help="Do not overlay ground-truth 2D keypoints.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/visualizations",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--out_name",
        default="multiview_pose_grid.png",
        help="Output image file name.",
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        nargs=2,
        default=(400, 400),
        metavar=("WIDTH", "HEIGHT"),
        help="Size of each sub-image in the grid.",
    )
    args = parser.parse_args()

    sample = np.load(args.sample, allow_pickle=True)
    pose3d_pred = None
    if args.pose3d:
        pose3d_pred = np.load(args.pose3d)

    image = render_multiview_grid(
        sample,
        frame=args.frame,
        pose3d_pred=pose3d_pred,
        views=args.views,
        show_gt_keypoints=not args.no_gt_keypoints,
        canvas_size=tuple(args.canvas_size),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    image.save(out_path)
    print(f"Saved multi-view visualization to {out_path}")


if __name__ == "__main__":
    main()
