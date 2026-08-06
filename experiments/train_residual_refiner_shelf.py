"""Train a residual refiner on top of DLT for Shelf.

The model takes the DLT triangulated skeleton and the multi-view 2D
observations, and predicts a residual 3D correction to reduce reprojection
error and suppress outlier frames.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_residual_refiner_shelf.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader
from motionflow_mv.fusion.residual_refiner import ResidualRefinerModel


DATA_ROOT = "tmp/voxelpose-pytorch/data/Shelf"
PICKLE_PATH = "outputs/shelf_matched_dataset.pkl"


def collate_dataset(dataset: dict, camera_ids):
    """Convert the matched dataset dict into batched tensors.

    Returns:
        inputs:  (N, V, J, 3)  -- pixel coordinates (x, y) + confidence
        targets: (N, J, 3)      -- DLT triangulated 3D skeleton
    """
    frames = sorted(dataset.keys())
    inputs, targets = [], []
    for fid in frames:
        item = dataset[fid]
        points_2d = item["points_2d"]  # (V, J, 2) pixels
        confidence = item["input"][..., 2]  # (V, J)
        inp = np.concatenate([points_2d, confidence[..., None]], axis=-1)
        inputs.append(inp)
        targets.append(item["target_3d"])
    return (
        torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32),
        torch.tensor(np.stack(targets, axis=0), dtype=torch.float32),
    )


def reprojection_loss(pred_3d, points_2d, proj_matrices, confidences):
    """Compute unweighted mean per-joint reprojection error (L2 distance in pixels).

    This matches the metric used in compare_all_shelf.py.
    """
    B, J = pred_3d.shape[:2]
    V = proj_matrices.shape[0]
    X_h = torch.cat([pred_3d, torch.ones(B, J, 1, device=pred_3d.device)], dim=-1)

    loss = 0.0
    for v in range(V):
        P = proj_matrices[v]
        x_h = torch.einsum("ik,bjk->bji", P, X_h)
        x = x_h[..., :2] / (x_h[..., 2:3] + 1e-6)
        diff = x - points_2d[:, v]
        loss += diff.norm(dim=-1).sum()

    return loss / (B * V * J)


def evaluate(model, loader, proj_matrices, device):
    """Return mean reprojection error (pixels) and 3D error (mm)."""
    model.eval()
    reproj_errors = []
    mse_3d_errors = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb, yb)
            loss = reprojection_loss(pred, xb[..., :2], proj_matrices, xb[..., 2])
            reproj_errors.append(loss.item())
            mse_3d_errors.append(torch.mean((pred - yb) ** 2).item())
    return np.mean(reproj_errors), np.mean(mse_3d_errors)


def main():
    parser = argparse.ArgumentParser(description="Train ResidualRefinerModel on Shelf.")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--residual_weight", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="outputs/residual_refiner_shelf.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = VoxelPoseShelfLoader(DATA_ROOT)
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    proj_matrices = torch.tensor(
        np.stack([cam.projection_matrix for cam in cameras], axis=0),
        dtype=torch.float32,
        device=device,
    )
    print(f"Loaded {len(cameras)} cameras: {camera_ids}")

    with open(PICKLE_PATH, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")
    print(f"Loaded {len(dataset)} matched frames")

    inputs, targets = collate_dataset(dataset, camera_ids)
    n_total = inputs.shape[0]
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_inputs, train_targets = inputs[:n_train], targets[:n_train]
    val_inputs, val_targets = inputs[n_train:], targets[n_train:]
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_inputs, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_inputs, val_targets),
        batch_size=args.batch_size,
    )

    model = ResidualRefinerModel(j=17, d=args.d, n_views=len(cameras)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, yb)
            reproj = reprojection_loss(pred, xb[..., :2], proj_matrices, xb[..., 2])
            res = pred - yb
            residual_reg = (res ** 2).mean()
            loss = reproj + args.residual_weight * residual_reg
            loss.backward()
            optimizer.step()
            train_loss += reproj.item() * xb.size(0)
        train_loss /= n_train

        val_reproj, val_mse = evaluate(model, val_loader, proj_matrices, device)
        scheduler.step(val_reproj)

        if val_reproj < best_val:
            best_val = val_reproj
            torch.save(model.state_dict(), output_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: train_reproj={train_loss:.2f}px, "
                f"val_reproj={val_reproj:.2f}px, val_3d_mse={val_mse:.2f}"
            )

    print(f"Best val reprojection error: {best_val:.2f}px")
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
