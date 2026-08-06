"""Evaluate a trained RayAttentionFusionModelV4 checkpoint on a canonical .npz."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v4_model import RayAttentionFusionModelV4


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        self.x = torch.from_numpy(data["points_2d"]).float()
        self.conf = torch.from_numpy(data["confidences"]).float()
        self.y = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).repeat(len(self.x), 1, 1, 1)
        self.R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).repeat(len(self.x), 1, 1, 1)
        self.t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).repeat(len(self.x), 1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.cat([self.x[idx], self.conf[idx, ..., None]], dim=-1)
        return x, self.y[idx], self.K[idx], self.R[idx], self.t[idx]


def collate_fn(batch):
    xb = torch.stack([b[0] for b in batch], dim=0)
    yb = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return xb, yb, K, R, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    model = RayAttentionFusionModelV4(j=j, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    loader = torch.utils.data.DataLoader(
        CameraDataset(data), batch_size=args.batch_size, collate_fn=collate_fn
    )

    mpjpe_clean = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            mpjpe_clean += err.item() * xb.size(0)
            count += xb.size(0)
    mpjpe_clean /= count
    print(f"MPJPE: {mpjpe_clean:.4f}m ({mpjpe_clean*1000:.1f}mm)")


if __name__ == "__main__":
    main()
