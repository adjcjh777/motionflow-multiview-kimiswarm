"""Evaluate a trained RayAttentionFusionModelV4Residual checkpoint on MPI-INF-3DHP.

Usage
-----
    conda run -n mf python experiments/eval_ray_attention_v4_residual_mpiinf3dhp.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/ray_attention_v4_residual_mpiinf3dhp.pth \
        --d 64 --batch_size 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v4_residual_model import RayAttentionFusionModelV4Residual
from motionflow_mv.fusion.ray_attention_v4_model import RayAttentionFusionModelV4


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.n_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        x = torch.cat(
            [self.points_2d[idx], self.confidences[idx].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.joints_3d[idx], self.K, self.R, self.t


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return xb, yb, K, R, t


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    loader = torch.utils.data.DataLoader(
        FrameDataset(args.dataset), batch_size=args.batch_size, collate_fn=collate_fn
    )

    model = RayAttentionFusionModelV4Residual(j=j, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    mpjpe = evaluate(model, loader, device)
    print(f"V4Residual MPJPE: {mpjpe:.4f}m ({mpjpe*1000:.2f}mm)")

    if args.baseline_checkpoint:
        baseline = RayAttentionFusionModelV4(j=j, d=args.d, n_views=n_views).to(device)
        baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True))
        baseline_mpjpe = evaluate(baseline, loader, device)
        print(f"V4 baseline MPJPE: {baseline_mpjpe:.4f}m ({baseline_mpjpe*1000:.2f}mm)")


if __name__ == "__main__":
    main()
