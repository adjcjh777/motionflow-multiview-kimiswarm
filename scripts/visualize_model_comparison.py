#!/usr/bin/env python3
"""Compare multi-view pose estimators on the same validation clip.

This script triangulates 2D keypoints with a DLT baseline and, optionally, one
or more trained model checkpoints, then renders diagnostic comparisons on a
common validation clip.  It is designed to help diagnose gaps between the v25
small baseline and local RTX 4090 runs by showing per-frame, per-joint, and
joint-frame error patterns side-by-side.

Smoke mode (no checkpoint, no real data) synthesises a short 4-view clip so
that the rendering pipeline can be verified on CPU in a few seconds.

Outputs
-------
Writes to ``--out_dir``::

    summary.json              – scalar metrics per method
    per_frame_mpjpe.png     – MPJPE time series
    per_joint_mpjpe.png     – per-joint MPJPE bar chart
    error_heatmap.png       – per-joint per-frame error heatmap

Examples
--------
    # CPU smoke test
    python scripts/visualize_model_comparison.py --smoke

    # Compare DLT vs a single checkpoint on a WebBridge clip
    python scripts/visualize_model_comparison.py \
        --dataset data/webbridge/h36m_meters/s_01_acts_02_multiview_m.npz \
        --checkpoint outputs/v25_small/best.pth \
        --model_class motionflow_mv.fusion.multiview_geometry_fusion_v25.MultiViewGeometryFusionV25 \
        --label v25_small
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe_batch, pa_mpjpe, per_joint_mpjpe
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for visualize_model_comparison.py. "
        "Install it with: pip install matplotlib"
    ) from exc


# ---------------------------------------------------------------------------
# Skeleton definitions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Camera / projection helpers
# ---------------------------------------------------------------------------
def _make_circular_cameras(n_views: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return synthetic intrinsics, rotations, translations for a circular rig."""
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    return (
        torch.from_numpy(np.stack(K_list, axis=0)).float(),
        torch.from_numpy(np.stack(R_list, axis=0)).float(),
        torch.from_numpy(np.stack(t_list, axis=0)).float(),
    )


def _project_points(
    joints_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Project (T, J, 3) joints through V cameras -> (T, V, J, 2)."""
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,tjb->vtja", R, joints_3d) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


def _build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build (V, 3, 4) projection matrix from K, R, t.

    Uses the standard pinhole projection P = K @ [R | t].
    """
    V = K.shape[0]
    P = np.zeros((V, 3, 4), dtype=np.float64)
    for v in range(V):
        P[v, :, :3] = K[v] @ R[v]
        P[v, :, 3] = K[v] @ t[v]
    return P


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_real_dataset(path: str):
    data = np.load(path)
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Dataset {path} missing keys: {missing}")
    return {
        "points_2d": torch.from_numpy(data["points_2d"]).float(),
        "confidences": torch.from_numpy(data["confidences"]).float(),
        "joints_3d": torch.from_numpy(data["joints_3d"]).float(),
        "camera_K": torch.from_numpy(data["camera_K"]).float(),
        "camera_R": torch.from_numpy(data["camera_R"]).float(),
        "camera_t": torch.from_numpy(data["camera_t"]).float(),
    }


def _make_synthetic_dataset(
    n_views: int = 4,
    n_frames: int = 60,
    n_joints: int = 17,
    noise_std: float = 0.5,
):
    """Create a synthetic (T, V, J, 2/3) clip for smoke testing."""
    K, R, t = _make_circular_cameras(n_views)
    torch.manual_seed(42)
    joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
    # Smooth temporally.
    for _ in range(2):
        joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

    points_2d = _project_points(joints_3d, K, R, t)
    if noise_std > 0:
        points_2d = points_2d + torch.randn_like(points_2d) * noise_std

    confidences = torch.ones(points_2d.shape[:-1])
    return {
        "points_2d": points_2d,
        "confidences": confidences,
        "joints_3d": joints_3d,
        "camera_K": K,
        "camera_R": R,
        "camera_t": t,
    }


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------
def _dlt_predict(points_2d: torch.Tensor, P: np.ndarray, weights: torch.Tensor) -> torch.Tensor:
    """Triangulate (T, V, J, 2) -> (T, J, 3) using DLT."""
    T, V, J, _ = points_2d.shape
    device = points_2d.device
    P = torch.from_numpy(P).to(device).float()
    pred = torch.zeros(T, J, 3, device=device, dtype=torch.float32)
    for ti in range(T):
        for ji in range(J):
            pred[ti, ji] = triangulate_dlt_torch(
                points_2d[ti, :, ji, :], P, weights[ti, :, ji]
            )
    return pred


def _load_model(model_class_path: str, checkpoint_path: str, n_views: int, n_joints: int):
    """Dynamically import a model class and load its checkpoint."""
    module_name, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    # Heuristic defaults; many ray-attention variants accept these arguments.
    model = model_cls(
        j=n_joints,
        d=64,
        n_views=n_views,
        n_heads=4,
        n_st_layers=2,
        residual_hidden=128,
    )
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Checkpoint load: missing keys {missing[:10]} ...")
        if unexpected:
            print(f"Checkpoint load: unexpected keys ignored {unexpected[:10]} ...")
    return model


def _model_predict(model: torch.nn.Module, points_2d: torch.Tensor) -> torch.Tensor:
    """Run a model on (T, V, J, 2) and return (T, J, 3).

    This wrapper only handles the simplest case where the model returns a tuple
    and the first element is (T, J, 3).  For more complex signatures, callers
    should extend this function.
    """
    # Most models expect (B, T, V, J, C).  Add a dummy batch dimension.
    x = points_2d[None, ...]
    with torch.no_grad():
        out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 4:
        out = out[0]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # Inputs are expected in metres; metrics report in millimetres.
    pred_mm = pred * 1000.0
    gt_mm = gt * 1000.0
    return {
        "mpjpe_mm": mpjpe_batch(pred_mm, gt_mm),
        "pa_mpjpe_mm": pa_mpjpe(pred_mm, gt_mm),
    }


def _per_frame_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_mm = pred * 1000.0
    gt_mm = gt * 1000.0
    return np.linalg.norm(pred_mm - gt_mm, axis=-1).mean(axis=-1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_per_frame_mpjpe(per_frame_errors, out_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    for label, errs in per_frame_errors.items():
        plt.plot(errs, label=label, alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("MPJPE (mm)")
    plt.title("Per-frame MPJPE comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig(out_path)


def _plot_per_joint_mpjpe(per_joint_errors, out_path: Path) -> None:
    x = np.arange(len(JOINT_NAMES))
    width = 0.8 / len(per_joint_errors)
    plt.figure(figsize=(14, 5))
    for i, (label, errs) in enumerate(per_joint_errors.items()):
        plt.bar(x + i * width, errs, width=width, label=label, alpha=0.8)
    plt.xticks(x, JOINT_NAMES, rotation=45, ha="right")
    plt.ylabel("MPJPE (mm)")
    plt.title("Per-joint MPJPE comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _savefig(out_path)


def _plot_error_heatmaps(per_joint_frame_errors, out_path: Path) -> None:
    n = len(per_joint_frame_errors)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (label, errs) in zip(axes, per_joint_frame_errors.items()):
        im = ax.imshow(errs.T, aspect="auto", cmap="inferno")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Joint index")
        ax.set_title(label)
        ax.set_yticks(range(len(JOINT_NAMES)))
        ax.set_yticklabels(JOINT_NAMES, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multi-view pose estimators on a common clip."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a canonical .npz dataset (required unless --smoke).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a CPU smoke test with synthetic data.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint to compare against DLT.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="motionflow_mv.fusion.ray_attention_v3_model.RayAttentionFusionModelV3",
        help="Full Python path to the model class to load.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="model",
        help="Label for the optional model checkpoint in plots/JSON.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/model_comparison",
        help="Directory to write figures and summary.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.smoke or args.dataset is None:
        print("Smoke mode: using synthetic dataset")
        dataset = _make_synthetic_dataset()
    else:
        dataset = _load_real_dataset(args.dataset)

    points_2d = dataset["points_2d"]
    confidences = dataset["confidences"]
    joints_3d_gt = dataset["joints_3d"]
    K = dataset["camera_K"]
    R = dataset["camera_R"]
    t = dataset["camera_t"]

    n_frames, n_views, n_joints = points_2d.shape[:3]
    print(f"Loaded clip: {n_frames} frames, {n_views} views, {n_joints} joints")

    P = _build_projection_matrix(K.numpy(), R.numpy(), t.numpy())

    # ------------------------------------------------------------------
    # Baseline DLT predictions
    # ------------------------------------------------------------------
    pred_dlt = _dlt_predict(points_2d, P, confidences)

    predictions = {"DLT": pred_dlt.numpy()}
    labels = ["DLT"]

    # ------------------------------------------------------------------
    # Optional model checkpoint
    # ------------------------------------------------------------------
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = _load_model(args.model_class, args.checkpoint, n_views, n_joints).to(device)
        model.eval()
        points_2d_device = points_2d.to(device)
        with torch.no_grad():
            pred_model = _model_predict(model, points_2d_device)
        predictions[args.label] = pred_model.cpu().numpy()
        labels.append(args.label)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    gt_arr = joints_3d_gt.numpy()
    summary: Dict[str, Any] = {}
    per_frame_errors = {}
    per_joint_errors = {}
    per_joint_frame_errors = {}

    for label in labels:
        pred = predictions[label]
        summary[label] = _compute_metrics(pred, gt_arr)
        per_frame_errors[label] = _per_frame_mpjpe(pred, gt_arr)
        per_joint_errors[label] = per_joint_mpjpe(pred * 1000.0, gt_arr * 1000.0)
        # (T, J) joint-frame errors in mm.
        per_joint_frame_errors[label] = np.linalg.norm((pred - gt_arr) * 1000.0, axis=-1)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_per_frame_mpjpe(per_frame_errors, out_dir / "per_frame_mpjpe.png")
    _plot_per_joint_mpjpe(per_joint_errors, out_dir / "per_joint_mpjpe.png")
    _plot_error_heatmaps(per_joint_frame_errors, out_dir / "error_heatmap.png")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary["n_frames"] = n_frames
    summary["n_views"] = n_views
    summary["n_joints"] = n_joints

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFigures and summary saved to: {out_dir}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
