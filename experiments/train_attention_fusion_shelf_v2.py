"""Train AttentionFusionV2 on real Shelf/VoxelPose data with camera parameters.

Usage:
    .venv/bin/python experiments/train_attention_fusion_shelf_v2.py \
        --data_root data/Shelf \
        --frame_start 300 --frame_end 600 \
        --epochs 200 --lr 1e-3 --d 64
"""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.attention_model_v2 import AttentionFusionModelV2


def load_precomputed_dataset(pkl_path: Path, frame_start: int, frame_end: int):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    inputs, targets = [], []
    for frame_idx in range(frame_start, frame_end + 1):
        if frame_idx not in data:
            continue
        item = data[frame_idx]
        inputs.append(torch.tensor(item["input"], dtype=torch.float32))
        targets.append(torch.tensor(item["target_3d"] / 1000.0, dtype=torch.float32))
    return torch.stack(inputs), torch.stack(targets)


def main():
    parser = argparse.ArgumentParser(description="Train AttentionFusionV2 on Shelf data.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)  # (V, 3, 4)
    proj_flat = proj_matrices.reshape(len(cameras), 12) / 1000.0  # (V, 12)

    precomputed_path = Path("outputs/shelf_matched_dataset.pkl")
    if not precomputed_path.exists():
        raise FileNotFoundError("Run prepare_shelf_dataset.py first.")
    inputs, targets = load_precomputed_dataset(precomputed_path, args.frame_start, args.frame_end)
    print(f"Dataset: {inputs.shape[0]} frames, {inputs.shape[1]} views, {inputs.shape[2]} joints")

    # Broadcast camera projection matrices for every sample
    n_samples = inputs.shape[0]
    camera_inputs = torch.tensor(np.tile(proj_flat, (n_samples, 1, 1)), dtype=torch.float32)

    # Simple frame-level split
    n = inputs.shape[0]
    n_val = int(n * args.val_ratio)
    train_inputs, val_inputs = inputs[n_val:], inputs[:n_val]
    train_targets, val_targets = targets[n_val:], targets[:n_val]
    train_cameras, val_cameras = camera_inputs[n_val:], camera_inputs[:n_val]

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_cameras, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_cameras, val_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    n_views = inputs.shape[1]
    j = inputs.shape[2]
    model = AttentionFusionModelV2(j=j, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, cb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mpjpe = 0.0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                pred = model(xb, cb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "attention_fusion_shelf_v2.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'attention_fusion_shelf_v2.pth'}")


if __name__ == "__main__":
    main()
