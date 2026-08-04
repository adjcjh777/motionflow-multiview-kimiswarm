"""Visualise residual corrections of RayAttentionFusionModelTemporalResidual.

For each joint in a selected clip, this script plots:
  1. the 3D trajectory of the raw DLT triangulated pose,
  2. the residual-corrected pose,
  3. the ground-truth 3D pose.

Usage
-----
    conda run -n mf python experiments/visualize_residual_corrections_v1.py \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
        --output_dir outputs/visualize_residual \
        --num_frames 100
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model_v3 import (
    RayAttentionFusionModelTemporalResidualV3,
)


def compute_mpjpe(pred, gt):
    """ pred, gt: (T, J, 3) numpy arrays. """
    return np.linalg.norm(pred - gt, axis=-1).mean()


def plot_joint_trajectory(ax3d, raw, refined, gt, joint_idx, title=""):
    """Plot 3D trajectory of raw, refined and GT for one joint."""
    for arr, color, label in [(raw, "r", "raw DLT"), (refined, "b", "residual-corrected"), (gt, "g", "ground truth")]:
        ax3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, alpha=0.7, label=label)
        ax3d.scatter(*arr[0, :], color=color, s=50, marker="o")
        ax3d.scatter(*arr[-1, :], color=color, s=50, marker="s")

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(title)
    ax3d.legend(loc="upper right", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description="Visualise residual corrections")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="outputs/visualize_residual")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to visualise")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame to use")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data.
    data = np.load(args.val)
    points_2d = data["points_2d"]  # (T, V, J, 2)
    confidences = data["confidences"]  # (T, V, J)
    joints_3d = data["joints_3d"]  # (T, J, 3)
    K_np = data["camera_K"]  # (V, 3, 3)
    R_np = data["camera_R"]  # (V, 3, 3)
    t_np = data["camera_t"]  # (V, 3)

    n_views = K_np.shape[0]
    j = points_2d.shape[2]
    total_frames = points_2d.shape[0]

    start = min(args.start_frame, max(0, total_frames - args.num_frames))
    # The temporal transformer has a fixed maximum length.
    max_len = 256
    end = min(start + args.num_frames, total_frames, start + max_len)
    num_frames = end - start

    x = np.concatenate([points_2d[start:end], confidences[start:end, ..., None]], axis=-1)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1, T, V, J, 3)
    gt = joints_3d[start:end]  # (T, J, 3)
    K = torch.from_numpy(K_np).float().to(device)
    R = torch.from_numpy(R_np).float().to(device)
    t = torch.from_numpy(t_np).float().to(device)

    # Build model.
    model = RayAttentionFusionModelTemporalResidualV3(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    with torch.no_grad():
        pred, _, pred_raw = model(x, K=K, R=R, t=t, return_raw=True)

    pred = pred.squeeze(0).cpu().numpy()  # (T, J, 3)
    pred_raw = pred_raw.squeeze(0).cpu().numpy()  # (T, J, 3)

    raw_err = compute_mpjpe(pred_raw, gt)
    refined_err = compute_mpjpe(pred, gt)
    print(f"Frames {start}:{end} | raw MPJPE = {raw_err*1000:.2f} mm | refined MPJPE = {refined_err*1000:.2f} mm")

    # Summary bar chart.
    per_joint_raw = np.linalg.norm(pred_raw - gt, axis=-1).mean(axis=0)  # (J,)
    per_joint_refined = np.linalg.norm(pred - gt, axis=-1).mean(axis=0)  # (J,)

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = np.arange(j)
    width = 0.35
    ax.bar(x_pos - width/2, per_joint_raw * 1000, width, label="raw DLT", color="r", alpha=0.7)
    ax.bar(x_pos + width/2, per_joint_refined * 1000, width, label="residual-corrected", color="b", alpha=0.7)
    ax.set_xlabel("Joint index")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title(f"Per-joint error (frames {start}-{end})")
    ax.set_xticks(x_pos)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_per_joint_mpjpe.png", dpi=150)
    plt.close(fig)

    # Per-joint plots.
    for joint_idx in range(j):
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(3, 3)
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        plot_joint_trajectory(ax3d, pred_raw[:, joint_idx], pred[:, joint_idx], gt[:, joint_idx], joint_idx, title=f"Joint {joint_idx} 3D trajectory")

        coords = [("X", 0), ("Y", 1), ("Z", 2)]
        for ax_idx, (name, coord) in enumerate(coords):
            ax = fig.add_subplot(gs[ax_idx, 1:])
            frames = np.arange(num_frames)
            ax.plot(frames, pred_raw[:, joint_idx, coord], color="r", alpha=0.7, label="raw DLT")
            ax.plot(frames, pred[:, joint_idx, coord], color="b", alpha=0.7, label="residual-corrected")
            ax.plot(frames, gt[:, joint_idx, coord], color="g", alpha=0.7, label="ground truth")
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{name} (m)")
            ax.set_title(f"Joint {joint_idx} - {name} coordinate over time")
            ax.legend(loc="upper right", fontsize=7)

        fig.tight_layout()
        fig.savefig(out_dir / f"joint_{joint_idx:02d}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {j} per-joint visualisations to {out_dir}")


if __name__ == "__main__":
    main()
