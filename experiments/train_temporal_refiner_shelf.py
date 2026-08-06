"""Train a temporal refiner over a sliding window of DLT outputs.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_temporal_refiner_shelf.py
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
from motionflow_mv.fusion.temporal_refiner import TemporalRefinerModel


DATA_ROOT = "tmp/voxelpose-pytorch/data/Shelf"
PICKLE_PATH = "outputs/shelf_matched_dataset.pkl"


def make_windowed_dataset(dataset: dict, window: int = 5):
    frames = sorted(dataset.keys())
    half = window // 2
    X, B, C = [], [], []
    for i in range(half, len(frames) - half):
        w = frames[i - half:i + half + 1]
        inputs = []
        baselines = []
        for f in w:
            item = dataset[f]
            points_2d = item["points_2d"]
            conf = item["input"][..., 2]
            inputs.append(np.concatenate([points_2d, conf[..., None]], axis=-1))
            baselines.append(item["target_3d"])
        X.append(np.stack(inputs, axis=0))  # (T, V, J, 3)
        B.append(np.stack(baselines, axis=0))  # (T, J, 3)
        C.append(frames[i])  # center frame id
    return (
        torch.tensor(np.stack(X, axis=0), dtype=torch.float32),
        torch.tensor(np.stack(B, axis=0), dtype=torch.float32),
        np.array(C),
    )


def reprojection_loss(pred_3d, points_2d, proj_matrices):
    """Unweighted mean per-joint reprojection L2 error (pixels)."""
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
    model.eval()
    reproj_errors = []
    with torch.no_grad():
        for xb, bb, _ in loader:
            xb = xb.to(device)
            bb = bb.to(device)
            pred = model(xb, bb)
            # center frame loss
            center = xb.shape[1] // 2
            loss = reprojection_loss(pred, xb[:, center, ..., :2], proj_matrices)
            reproj_errors.append(loss.item())
    return np.mean(reproj_errors)


def main():
    parser = argparse.ArgumentParser(description="Train TemporalRefinerModel on Shelf.")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="outputs/temporal_refiner_shelf.pth")
    parser.add_argument("--pretrained", type=str, default=None)
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

    X, B, C = make_windowed_dataset(dataset, window=args.window)
    print(f"Windowed samples: {X.shape[0]}")

    n_total = X.shape[0]
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_X, train_B, train_C = X[:n_train], B[:n_train], C[:n_train]
    val_X, val_B, val_C = X[n_train:], B[n_train:], C[n_train:]
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X, train_B, torch.arange(n_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_X, val_B, torch.arange(n_val)),
        batch_size=args.batch_size,
    )

    model = TemporalRefinerModel(j=17, d=args.d, n_views=len(cameras), hidden=args.hidden).to(device)
    if args.pretrained is not None:
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, bb, _ in train_loader:
            xb = xb.to(device)
            bb = bb.to(device)
            optimizer.zero_grad()
            pred = model(xb, bb)
            center = xb.shape[1] // 2
            loss = reprojection_loss(pred, xb[:, center, ..., :2], proj_matrices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train

        val_reproj = evaluate(model, val_loader, proj_matrices, device)
        scheduler.step(val_reproj)

        if val_reproj < best_val:
            best_val = val_reproj
            torch.save(model.state_dict(), output_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: train_reproj={train_loss:.2f}px, val_reproj={val_reproj:.2f}px")

    print(f"Best val reprojection error: {best_val:.2f}px")
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
