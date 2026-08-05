"""Ablation: RayAttentionFusionModel (v1, view-only) vs. v2 (view + joint).

Trains both models on the same H36M subset and reports clean / corrupted MPJPE.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel
from motionflow_mv.fusion.ray_attention_v2_model import RayAttentionFusionModelV2


class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, idx: np.ndarray):
        self.x = torch.from_numpy(data["points_2d"][idx]).float()
        self.conf = torch.from_numpy(data["confidences"][idx]).float()
        self.y = torch.from_numpy(data["joints_3d"][idx]).float()
        if data["camera_K"].shape[0] == self.x.shape[0]:
            self.K = torch.from_numpy(data["camera_K"][idx]).float()
            self.R = torch.from_numpy(data["camera_R"][idx]).float()
            self.t = torch.from_numpy(data["camera_t"][idx]).float()
        else:
            self.K = torch.from_numpy(data["camera_K"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.R = torch.from_numpy(data["camera_R"]).float().unsqueeze(0).repeat(len(idx), 1, 1, 1)
            self.t = torch.from_numpy(data["camera_t"]).float().unsqueeze(0).repeat(len(idx), 1, 1)

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


def augment_batch(x, noise_std=0.5, dropout_rate=0.1, outlier_rate=0.02, outlier_scale=100.0):
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            total += (pred - yb).norm(dim=-1).sum().item()
            n += xb.size(0) * xb.shape[2]
    return total / n


def train_one(model, train_loader, val_loader, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        val_err = evaluate(model, val_loader, device)
        if val_err < best:
            best = val_err
        print(f"  Epoch {epoch}: val_MPJPE={val_err:.4f}mm")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/h36m_hf/s_01_train_subset_500.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n = data["joints_3d"].shape[0]
    rng = np.random.default_rng(2027)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.2))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_loader = torch.utils.data.DataLoader(
        CameraDataset(data, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        CameraDataset(data, val_idx), batch_size=args.batch_size, collate_fn=collate_fn
    )

    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    results = {}
    for name, Cls in [("v1_view_only", RayAttentionFusionModel), ("v2_view_joint", RayAttentionFusionModelV2)]:
        print(f"\n=== Training {name} ===")
        model = Cls(j=j, d=args.d, n_views=n_views).to(device)
        start = time.time()
        best = train_one(model, train_loader, val_loader, args.epochs, device)
        results[name] = {"best_val_mpjpe_mm": best, "time_s": time.time() - start}

    print("\n=== Ablation summary ===")
    for name, res in results.items():
        print(f"{name}: best_val_MPJPE={res['best_val_mpjpe_mm']:.4f}mm, time={res['time_s']:.1f}s")


if __name__ == "__main__":
    main()
