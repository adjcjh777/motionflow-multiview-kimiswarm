"""Train ray-aware attention fusion on synthetic SMPL multi-view data.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_ray_attention_synthetic.py \
        --dataset outputs/synthetic_multiview_dataset.npz \
        --epochs 50 --lr 1e-3 --d 64 --batch_size 32
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel


class CameraDataset(torch.utils.data.Dataset):
    """Dataset that yields per-sample cameras alongside points and targets."""

    def __init__(self, data: dict, idx: np.ndarray):
        self.x = torch.from_numpy(data["points_2d"][idx]).float()
        self.conf = torch.from_numpy(data["confidences"][idx]).float()
        self.y = torch.from_numpy(data["joints_3d"][idx]).float()
        self.K = torch.from_numpy(data["camera_K"][idx]).float()
        self.R = torch.from_numpy(data["camera_R"][idx]).float()
        self.t = torch.from_numpy(data["camera_t"][idx]).float()

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
    parser.add_argument("--dataset", type=str, default="outputs/synthetic_multiview_dataset.npz")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.dataset)
    n = data["joints_3d"].shape[0]
    n_val = int(n * args.val_ratio)
    perm = np.random.permutation(n)

    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_dataset = CameraDataset(data, train_idx)
    val_dataset = CameraDataset(data, val_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    n_views = data["camera_K"].shape[1]
    j = data["points_2d"].shape[2]
    model = RayAttentionFusionModel(j=j, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb, K, R, t in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                K, R, t = K.to(device), R.to(device), t.to(device)
                pred, _ = model(xb, K=K, R=R, t=t)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "ray_attention_synthetic.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}m")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'ray_attention_synthetic.pth'}")


if __name__ == "__main__":
    main()
