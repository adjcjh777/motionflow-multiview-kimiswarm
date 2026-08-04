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

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.ray_attention_model import RayAttentionFusionModel


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
    points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
    confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
    joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)

    # Build Camera objects
    camera_K = data["camera_K"]
    camera_R = data["camera_R"]
    camera_t = data["camera_t"]
    cameras = [Camera(K=camera_K[v], R=camera_R[v], t=camera_t[v]) for v in range(camera_K.shape[0])]

    # Build model input: (x, y, confidence)
    x = torch.cat([points_2d, confidences[..., None]], dim=-1)  # (T, V, J, 3)

    # Temporal shuffle
    n = x.shape[0]
    n_val = int(n * args.val_ratio)
    perm = torch.randperm(n)
    x = x[perm]
    joints_3d = joints_3d[perm]

    train_x, val_x = x[n_val:], x[:n_val]
    train_y, val_y = joints_3d[n_val:], joints_3d[:n_val]

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    n_views = x.shape[1]
    j = x.shape[2]
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
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb, cameras)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb, cameras)
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
