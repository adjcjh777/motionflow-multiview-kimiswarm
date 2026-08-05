"""Fine-tune synthetic-pretrained AttentionFusion on real Shelf data.

Usage:
    .venv/bin/python experiments/train_attention_fusion_shelf_finetune.py \
        --data_root data/Shelf \
        --pretrained outputs/attention_fusion_synthetic.pth \
        --frame_start 300 --frame_end 600
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
from motionflow_mv.fusion.attention_model import AttentionFusionModel


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
    parser = argparse.ArgumentParser(description="Fine-tune AttentionFusion on Shelf data.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pkl_path = Path("outputs/shelf_matched_dataset.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError("Run prepare_shelf_dataset.py first.")
    inputs, targets = load_precomputed_dataset(pkl_path, args.frame_start, args.frame_end)
    print(f"Dataset: {inputs.shape[0]} frames, {inputs.shape[1]} views, {inputs.shape[2]} joints")

    n = inputs.shape[0]
    n_val = int(n * args.val_ratio)
    train_inputs, val_inputs = inputs[n_val:], inputs[:n_val]
    train_targets, val_targets = targets[n_val:], targets[:n_val]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_inputs, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_inputs, val_targets),
        batch_size=args.batch_size,
    )

    n_views = inputs.shape[1]
    j = inputs.shape[2]
    model = AttentionFusionModel(j=j, d=args.d, n_views=n_views).to(device)
    model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()

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
            pred = model(xb)
            mse_loss = mse_criterion(pred, yb)
            mpjpe_loss = (pred - yb).norm(dim=-1).mean()
            loss = mse_loss + 0.1 * mpjpe_loss
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
                pred = model(xb)
                loss = mse_criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_mpjpe += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_mpjpe /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "attention_fusion_shelf_ft.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'attention_fusion_shelf_ft.pth'}")


if __name__ == "__main__":
    main()
