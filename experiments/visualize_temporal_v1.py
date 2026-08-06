"""Visualize temporal ray-attention 3D pose predictions vs ground truth.

Loads a trained ``RayAttentionFusionModelTemporal`` checkpoint (or a DLT
baseline if no checkpoint is supplied), runs inference on a contiguous clip
from an MPI-INF-3DHP canonical .npz, and renders:

1. Per-frame 3D scatter plots of predicted (red) vs ground-truth (blue) poses.
2. A GIF animation assembled from those per-frame plots.
3. 3D joint-trajectory traces over the clip for a few representative joints.
4. A per-frame MPJPE time-series plot.

Outputs are written to ``outputs/visualize_temporal/`` by default.

Usage
-----
    conda run -n mf python experiments/visualize_temporal_v1.py \\
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \\
        --checkpoint outputs/ray_attention_temporal_smoke.pth \\
        --start_frame 500 --clip_len 60

Dependencies
------------
    numpy, torch, matplotlib, PIL (Pillow; optional, for GIF output)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Use a non-interactive matplotlib backend so the script runs headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_model import RayAttentionFusionModelTemporal
from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize temporal ray-attention 3D pose predictions vs ground truth."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz",
        help="Path to a canonical .npz dataset (T, V, J).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ray_attention_temporal_smoke.pth",
        help="Path to a RayAttentionFusionModelTemporal .pth checkpoint. "
        "If the file does not exist, a confidence-weighted DLT baseline is used.",
    )
    parser.add_argument("--start_frame", type=int, default=500, help="First frame of the clip to visualize.")
    parser.add_argument("--clip_len", type=int, default=60, help="Number of frames in the clip.")
    parser.add_argument("--d", type=int, default=64, help="Model embedding dimension.")
    parser.add_argument("--n_temporal_layers", type=int, default=2, help="Number of temporal transformer layers.")
    parser.add_argument("--gif_fps", type=int, default=10, help="Frames per second for the output GIF.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualize_temporal",
        help="Directory where outputs are saved.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data / model helpers
# --------------------------------------------------------------------------- #
def load_dataset(path: str):
    """Load canonical .npz and return a dict of numpy arrays."""
    data = np.load(path)
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Dataset {path} missing keys: {missing}")
    return {
        "points_2d": data["points_2d"],
        "confidences": data["confidences"],
        "joints_3d": data["joints_3d"],
        "camera_K": data["camera_K"],
        "camera_R": data["camera_R"],
        "camera_t": data["camera_t"],
    }


def build_clip(dataset: dict, start: int, length: int):
    """Extract a contiguous clip and return (x, gt_3d, K, R, t)."""
    total = dataset["joints_3d"].shape[0]
    if start + length > total:
        length = total - start
        if length <= 0:
            raise ValueError(f"start_frame {start} is beyond the sequence length {total}")

    points_2d = dataset["points_2d"][start : start + length]
    confidences = dataset["confidences"][start : start + length]
    joints_3d = dataset["joints_3d"][start : start + length]

    # (T, V, J, 3) observation tensor.
    x = np.concatenate([points_2d, confidences[..., None]], axis=-1)

    K = torch.from_numpy(dataset["camera_K"]).float()
    R = torch.from_numpy(dataset["camera_R"]).float()
    t = torch.from_numpy(dataset["camera_t"]).float()

    return x, joints_3d, K, R, t, length


def load_model(checkpoint_path: str, j: int, n_views: int, d: int, n_temporal_layers: int, device: torch.device):
    """Load a RayAttentionFusionModelTemporal checkpoint."""
    model = RayAttentionFusionModelTemporal(
        j=j, d=d, n_views=n_views, n_temporal_layers=n_temporal_layers
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def dlt_baseline_clip(
    points_2d: torch.Tensor,
    confidences: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted DLT triangulation for a clip (T, V, J, 2/1)."""
    # Build projection matrices P = K [R | t].
    Rt = torch.cat([R, t[..., None]], dim=-1)  # (V, 3, 4)
    P = K @ Rt  # (V, 3, 4)

    T, V, J = confidences.shape
    pred = torch.zeros(T, J, 3, dtype=points_2d.dtype, device=points_2d.device)

    # triangulate_dlt_torch supports batch dimension; batch over frames.
    for j in range(J):
        # points: (T, V, 2), weights: (T, V)
        pred[:, j, :] = triangulate_dlt_torch(
            points_2d[:, :, j, :],
            P.unsqueeze(0).expand(T, -1, -1, -1),
            weights=confidences[:, :, j],
        )
    return pred


def run_inference(
    x: np.ndarray,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    checkpoint_path: str | None,
    d: int,
    n_temporal_layers: int,
    device: torch.Tensor,
) -> tuple:
    """Return (pred_3d (T,J,3), weights (T,V,J), used_checkpoint bool)."""
    T, V, J, _ = x.shape
    x_t = torch.from_numpy(x).float().to(device)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model = load_model(checkpoint_path, J, V, d, n_temporal_layers, device)
        with torch.no_grad():
            pred, weights = model(x_t.unsqueeze(0), K=K.to(device), R=R.to(device), t=t.to(device))
        pred = pred[0].cpu()
        weights = weights[0].cpu()
        return pred, weights, True

    points_2d = x_t[..., :2]
    confidences = x_t[..., 2]
    pred = dlt_baseline_clip(points_2d, confidences, K, R, t)
    weights = confidences  # (T, V, J)
    return pred, weights, False


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _set_equal_3d_limits(ax, coords_pred, coords_gt):
    """Compute common 3D limits with a small margin."""
    all_pts = np.concatenate([coords_pred.reshape(-1, 3), coords_gt.reshape(-1, 3)], axis=0)
    min_vals = all_pts.min(axis=0)
    max_vals = all_pts.max(axis=0)
    centers = (min_vals + max_vals) / 2.0
    half = (max_vals - min_vals).max() / 2.0
    if half <= 0:
        half = 1.0
    margin = half * 1.05
    ax.set_xlim(centers[0] - margin, centers[0] + margin)
    ax.set_ylim(centers[1] - margin, centers[1] + margin)
    ax.set_zlim(centers[2] - margin, centers[2] + margin)


def plot_frame_3d(pred_3d, gt_3d, output_path: Path, title: str = ""):
    """Render a single 3D scatter plot: predicted vs ground truth."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c="blue", s=40, alpha=0.7, label="Ground truth")
    ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c="red", s=40, alpha=0.7, label="Predicted")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("3D pose: predicted (red) vs ground truth (blue)")
    ax.legend(loc="upper right")

    _set_equal_3d_limits(ax, pred_3d, gt_3d)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def render_frame_sequence(pred_3d, gt_3d, output_dir: Path, fps: int = 10):
    """Save one 3D pose plot per frame and try to assemble a GIF."""
    T = pred_3d.shape[0]
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for t in range(T):
        path = frame_dir / f"frame_{t:04d}.png"
        plot_frame_3d(pred_3d[t], gt_3d[t], path, title=f"Frame {t}")
        frame_paths.append(path)

    # Assemble GIF if Pillow is available.
    try:
        from PIL import Image

        images = [Image.open(p) for p in frame_paths]
        gif_path = output_dir / "temporal_pose.gif"
        duration_ms = int(1000 / max(fps, 1))
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved temporal GIF to {gif_path}")
    except Exception as exc:
        print(f"GIF creation skipped: {exc}")

    return frame_paths


def plot_trajectories(pred_3d, gt_3d, output_path: Path, joint_indices=None):
    """Plot 3D trajectories of selected joints across the clip."""
    if joint_indices is None:
        # pelvis, right wrist, left ankle, head approximations
        joint_indices = [0, min(7, pred_3d.shape[1] - 1), min(10, pred_3d.shape[1] - 1), min(20, pred_3d.shape[1] - 1)]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_indices)))
    for i, j in enumerate(joint_indices):
        ax.plot(gt_3d[:, j, 0], gt_3d[:, j, 1], gt_3d[:, j, 2], color=colors[i], linestyle="-", alpha=0.6, linewidth=2)
        ax.plot(pred_3d[:, j, 0], pred_3d[:, j, 1], pred_3d[:, j, 2], color=colors[i], linestyle="--", alpha=0.8, linewidth=2)
        # Mark start points.
        ax.scatter(gt_3d[0, j, 0], gt_3d[0, j, 1], gt_3d[0, j, 2], color=colors[i], s=40, marker="o")
        ax.scatter(pred_3d[0, j, 0], pred_3d[0, j, 1], pred_3d[0, j, 2], color=colors[i], s=40, marker="x")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Joint trajectories across time (solid=GT, dashed=prediction)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved 3D trajectory plot to {output_path}")


def plot_mpjpe(pred_3d, gt_3d, output_path: Path):
    """Plot per-frame MPJPE in millimeters."""
    errors = np.linalg.norm(pred_3d - gt_3d, axis=-1) * 1000.0  # (T, J)
    per_frame = errors.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_frame, linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title(f"Per-frame mean per-joint position error (mean={per_frame.mean():.2f} mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved MPJPE time-series plot to {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_dataset(args.dataset)
    x, gt_3d, K, R, t, clip_len = build_clip(dataset, args.start_frame, args.clip_len)
    print(f"Loaded clip: {clip_len} frames starting at frame {args.start_frame}")

    checkpoint_path = args.checkpoint if Path(args.checkpoint).exists() else None
    if checkpoint_path is None and args.checkpoint:
        print(f"Checkpoint {args.checkpoint} not found; falling back to DLT baseline.")

    pred_3d, weights, used_checkpoint = run_inference(
        x, K, R, t, checkpoint_path, args.d, args.n_temporal_layers, device
    )
    pred_3d_np = pred_3d.numpy()
    gt_3d_np = gt_3d

    # Quantitative summary.
    errors_mm = np.linalg.norm(pred_3d_np - gt_3d_np, axis=-1) * 1000.0
    mpjpe = errors_mm.mean()
    print(f"MPJPE = {mpjpe:.4f} mm | per-frame range: [{errors_mm.mean(axis=1).min():.2f}, {errors_mm.mean(axis=1).max():.2f}] mm")
    print(f"Per-view weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-frame 3D plots + GIF.
    render_frame_sequence(pred_3d_np, gt_3d_np, output_dir, fps=args.gif_fps)

    # 2. Trajectory plots.
    plot_trajectories(pred_3d_np, gt_3d_np, output_dir / "joint_trajectories.png")

    # 3. MPJPE over time.
    plot_mpjpe(pred_3d_np, gt_3d_np, output_dir / "mpjpe_time.png")

    # Save a small summary file.
    summary = {
        "dataset": args.dataset,
        "start_frame": int(args.start_frame),
        "clip_len": int(clip_len),
        "checkpoint": str(checkpoint_path) if used_checkpoint else "DLT baseline",
        "mpjpe_mm": float(mpjpe),
        "mpjpe_per_frame_mm": errors_mm.mean(axis=1).tolist(),
    }
    np.savez(output_dir / "summary.npz", **summary)
    print(f"Saved summary to {output_dir / 'summary.npz'}")
    print(f"All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
