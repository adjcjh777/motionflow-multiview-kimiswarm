"""Train RobustTriangulationModel on Shelf DLT pseudo-labels.

The model learns per-view weights for differentiable DLT using real Shelf
2D keypoints and camera projection matrices.  Because no 3D GT is available,
the DLT triangulated skeleton is used as the pseudo-target.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_robust_triangulation_shelf.py
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
from motionflow_mv.fusion.robust_triangulation import RobustTriangulationModel


def reprojection_loss(pred_3d, points_2d, proj_matrices):
    """Mean per-joint reprojection error (L2 distance in pixels)."""
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


def main():
    parser = argparse.ArgumentParser(description="Train RobustTriangulationModel on Shelf.")
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="outputs/robust_triangulation_shelf.pth")
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

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    n_views = len(cameras)
    j = inputs.shape[2]
    model = RobustTriangulationModel(j=j, d=args.d, n_views=n_views).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_reproj = 0.0
        train_mse = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, proj_matrices)
            reproj = reprojection_loss(pred, xb[..., :2], proj_matrices)
            mse = torch.mean((pred - yb) ** 2)
            loss = reproj + 1e-4 * mse
            loss.backward()
            optimizer.step()
            train_reproj += reproj.item() * xb.size(0)
            train_mse += mse.item() * xb.size(0)
        train_reproj /= n_train
        train_mse /= n_train

        model.eval()
        val_reproj = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, proj_matrices)
                val_reproj += reprojection_loss(pred, xb[..., :2], proj_matrices).item() * xb.size(0)
                val_mse += torch.mean((pred - yb) ** 2).item() * xb.size(0)
        val_reproj /= n_val
        val_mse /= n_val

        if val_reproj < best_val:
            best_val = val_reproj
            torch.save(model.state_dict(), output_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: train_reproj={train_reproj:.2f}px, "
                f"val_reproj={val_reproj:.2f}px, val_mse={val_mse:.2f}"
            )

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_path}")


if __name__ == "__main__":
    main()
