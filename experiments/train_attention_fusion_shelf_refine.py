"""Train AttentionFusion to refine DLT output on Shelf data.

Loss is reprojection error of (dlt_3d + small_correction). The correction is
scaled so it stays small, and the model starts from a strong DLT baseline.

Usage:
    .venv/bin/python experiments/train_attention_fusion_shelf_refine.py \
        --data_root data/Shelf \
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
    inputs, targets, points_2d_list = [], [], []
    for frame_idx in range(frame_start, frame_end + 1):
        if frame_idx not in data:
            continue
        item = data[frame_idx]
        inputs.append(torch.tensor(item["input"], dtype=torch.float32))
        targets.append(torch.tensor(item["target_3d"] / 1000.0, dtype=torch.float32))
        points_2d_list.append(item["points_2d"])
    return torch.stack(inputs), torch.stack(targets), points_2d_list


def reprojection_loss(pred_3d, points_2d, proj_matrices):
    """ pred_3d: (B, J, 3) in mm, points_2d: (B, V, J, 2), proj_matrices: (V, 3, 4) """
    B, J, _ = pred_3d.shape
    V = points_2d.shape[1]
    X_h = torch.cat([pred_3d, torch.ones(B, J, 1, device=pred_3d.device)], dim=-1)
    X_h_t = X_h.permute(0, 2, 1).reshape(B * J, 4)
    x_h = torch.matmul(proj_matrices, X_h_t.T)
    x = x_h[:, :2, :] / (x_h[:, 2:3, :] + 1e-6)
    x = x.permute(2, 0, 1).reshape(B, V, J, 2)
    return torch.mean((x - points_2d) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Train AttentionFusion to refine DLT.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = VoxelPoseShelfLoader(args.data_root)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj_matrices = np.stack([cam.projection_matrix for cam in cameras], axis=0)
    proj_tensors = torch.tensor(proj_matrices, dtype=torch.float32).to(device)

    pkl_path = Path("outputs/shelf_matched_dataset.pkl")
    inputs, targets, points_2d_list = load_precomputed_dataset(
        pkl_path, args.frame_start, args.frame_end
    )
    points_2d_tensor = torch.tensor(np.stack(points_2d_list, axis=0), dtype=torch.float32)
    print(f"Dataset: {inputs.shape[0]} frames, {inputs.shape[1]} views, {inputs.shape[2]} joints")

    n = inputs.shape[0]
    n_val = int(n * args.val_ratio)
    train_inputs, val_inputs = inputs[n_val:], inputs[:n_val]
    train_targets, val_targets = targets[n_val:], targets[:n_val]
    train_points2d, val_points2d = points_2d_tensor[n_val:], points_2d_tensor[:n_val]

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets, train_points2d)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets, val_points2d)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    n_views = inputs.shape[1]
    j = inputs.shape[2]
    model = AttentionFusionModel(j=j, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, p2d in train_loader:
            xb, yb, p2d = xb.to(device), yb.to(device), p2d.to(device)
            optimizer.zero_grad()
            delta = model(xb) * 10.0  # scale to keep corrections small (10 mm)
            pred_3d = yb * 1000.0 + delta  # yb in meters -> mm, then add correction
            loss = reprojection_loss(pred_3d, p2d, proj_tensors)
            mse_reg = mse_criterion(delta, torch.zeros_like(delta))
            loss = loss + args.alpha * mse_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, p2d in val_loader:
                xb, yb, p2d = xb.to(device), yb.to(device), p2d.to(device)
                delta = model(xb) * 10.0
                pred_3d = yb * 1000.0 + delta
                loss = reprojection_loss(pred_3d, p2d, proj_tensors)
                mse_reg = mse_criterion(delta, torch.zeros_like(delta))
                loss = loss + args.alpha * mse_reg
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "attention_fusion_shelf_refine.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / 'attention_fusion_shelf_refine.pth'}")


if __name__ == "__main__":
    main()
