"""Generate paper figures for MotionFlow-MultiView.

Figures:
    1. 3D skeleton comparison (GT vs prediction) for a single frame.
    2. Per-joint MPJPE bar chart.
    3. Per-view weight heatmap.
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

from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


class TemporalClipDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def draw_skeleton_3d(ax, joints, color, label=None):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=20)
    if label:
        ax.set_title(label)


def generate_figures(checkpoint, dataset, output_dir, d=64, n_temporal_layers=2, residual_hidden=128):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    loader = torch.utils.data.DataLoader(
        TemporalClipDataset(dataset, 13), batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = RayAttentionFusionModelTemporalResidual(
        j=j, d=d, n_views=n_views, n_temporal_layers=n_temporal_layers, residual_hidden=residual_hidden
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    all_errs = []
    all_weights = []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, weights = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1)  # (B, T, J)
            all_errs.append(err.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            break  # only first clip for skeleton figure

    all_errs = np.concatenate(all_errs, axis=0)  # (T, J)
    per_joint_err = all_errs.mean(axis=0) * 1000  # mm

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111, projection="3d")
    # draw skeleton comparison for middle frame
    mid = pred.shape[1] // 2
    gt_joints = yb[0, mid].cpu().numpy()
    pred_joints = pred[0, mid].cpu().numpy()
    draw_skeleton_3d(ax, gt_joints, "blue", "GT")
    draw_skeleton_3d(ax, pred_joints, "red", "Pred")
    ax.set_title("3D skeleton comparison")
    fig.savefig(output_dir / "skeleton_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(j), per_joint_err)
    ax.set_xlabel("Joint index")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-joint MPJPE")
    fig.savefig(output_dir / "per_joint_mpjpe.png", dpi=150)
    plt.close(fig)

    weights = all_weights[0][0, 0]  # (V, J)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xlabel("Joint index")
    ax.set_ylabel("View index")
    ax.set_title("Per-view per-joint DLT weights")
    fig.colorbar(im, ax=ax)
    fig.savefig(output_dir / "weight_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/paper_figures")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    args = parser.parse_args()
    generate_figures(
        args.checkpoint, args.dataset, args.output_dir,
        args.d, args.n_temporal_layers, args.residual_hidden,
    )


if __name__ == "__main__":
    main()
