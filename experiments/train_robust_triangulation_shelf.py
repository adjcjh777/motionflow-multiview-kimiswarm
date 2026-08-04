"""Train the confidence-weighted DLT triangulation model on Shelf data.

The model predicts per-view weights and triangulates 3D joints in a fully
differentiable way.  We train it on the matched Shelf dataset created by
``run_shelf_voxelpose_baseline.py`` using reprojection loss.

Usage:
    /d/anaconda3/envs/mf/python.exe experiments/train_robust_triangulation_shelf.py
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
    """Compute confidence-weighted reprojection error.

    Args:
        pred_3d:    (B, J, 3)
        points_2d:  (B, V, J, 2) pixel coordinates
        proj_matrices: (V, 3, 4)
        confidences: (B, V, J)

    Returns:
        Scalar loss.
    """
    B, J = pred_3d.shape[:2]
    V = proj_matrices.shape[0]
    # Homogeneous 3D points
    X_h = torch.cat([pred_3d, torch.ones(B, J, 1, device=pred_3d.device)], dim=-1)

    loss = 0.0
    total_weight = 0.0
    for v in range(V):
        P = proj_matrices[v]  # (3, 4)
        x_h = torch.einsum("ik,bjk->bji", P, X_h)  # (B, J, 3)
        x = x_h[..., :2] / (x_h[..., 2:3] + 1e-6)
        diff = x - points_2d[:, v]  # (B, J, 2)
        w = confidences[:, v]  # (B, J)
        loss += (w * (diff ** 2).sum(dim=-1)).sum()
        total_weight += w.sum()

    return loss / (total_weight + 1e-8)


def evaluate(model, loader, proj_matrices, device):
    """Return mean reprojection error (pixels) and 3D error (mm)."""
    model.eval()
    reproj_errors = []
    mse_3d_errors = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb, proj_matrices)
            # Reprojection loss
            loss = reprojection_loss(pred, xb[..., :2], proj_matrices, xb[..., 2])
            reproj_errors.append(loss.item())
            # 3D MSE against DLT target
            mse_3d_errors.append(torch.mean((pred - yb) ** 2).item())
    return np.mean(reproj_errors), np.mean(mse_3d_errors)


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

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_inputs, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_inputs, val_targets),
        batch_size=args.batch_size,
    )

    model = RobustTriangulationModel(j=17, d=args.d, n_views=len(cameras)).to(device)
    # Initialize close to equal weights (sigmoid(2.0) ~ 0.88)
    with torch.no_grad():
        nn.init.zeros_(model.weight_head.weight)
        nn.init.constant_(model.weight_head.bias, 2.0)

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
            pred = model(xb, proj_matrices)
            loss = reprojection_loss(pred, xb[..., :2], proj_matrices, xb[..., 2])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
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
