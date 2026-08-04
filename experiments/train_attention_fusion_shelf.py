"""Train AttentionFusion on real Shelf/Campus/VoxelPose data using DLT pseudo-GT.

This script converts matched 2D predictions into training samples for the
AttentionFusion model.  Because 3D ground truth is not available, the
triangulated DLT result is used as a pseudo-target.

Usage:
    /d/anaconda3/envs/jz_py310/python.exe experiments/train_attention_fusion_shelf.py \
        --data_root tmp/voxelpose-pytorch/data/Shelf \
        --dataset shelf --frame_start 300 --frame_end 600
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

from motionflow_mv.data.voxelpose_loader import VoxelPoseShelfLoader, VoxelPoseCampusLoader
from motionflow_mv.pipeline import MultiViewPipeline
from motionflow_mv.pipeline_utils import select_best_person_group
from motionflow_mv.fusion.attention_model import AttentionFusionModel


def build_dataset(loader, frame_start: int, frame_end: int):
    """Return (inputs, targets) tensors for matched frames.

    inputs:  (N, V, J, 3)  -> (x, y, confidence)
    targets: (N, J, 3)     -> DLT triangulated 3D
    """
    camera_ids = sorted(loader.cameras.keys(), key=lambda x: int(x))
    cameras = [loader.get_camera(cid) for cid in camera_ids]
    pipeline = MultiViewPipeline(estimator=None)

    inputs, targets = [], []
    for frame_idx in range(frame_start, frame_end + 1):
        frame_predictions = {cid: loader.get_frame_predictions(cid, frame_idx) for cid in camera_ids}
        if any(len(p) == 0 for p in frame_predictions.values()):
            continue

        try:
            _, points_2d, confidences = select_best_person_group(
                frame_predictions, loader.cameras, camera_ids
            )
        except ValueError:
            continue

        # Build pseudo-GT with DLT
        pred_3d = pipeline.fuse_frame(points_2d, confidences, cameras)

        # Input: (V, J, 3); normalize 2D and 3D to ~meter scale for stability
        points_2d_norm = points_2d / 1000.0
        pred_3d_norm = pred_3d / 1000.0
        inp = np.concatenate([points_2d_norm, confidences[..., None]], axis=-1)
        inputs.append(torch.tensor(inp, dtype=torch.float32))
        targets.append(torch.tensor(pred_3d_norm, dtype=torch.float32))

    if not inputs:
        raise ValueError("No valid frames found in the specified range.")

    return torch.stack(inputs), torch.stack(targets)


def load_precomputed_dataset(pkl_path: Path, frame_start: int, frame_end: int):
    """Load matched frames from prepare_shelf_dataset.py output."""
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
    parser = argparse.ArgumentParser(description="Train AttentionFusion on Shelf/Campus data.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="shelf", choices=["shelf", "campus"])
    parser.add_argument("--frame_start", type=int, default=300)
    parser.add_argument("--frame_end", type=int, default=600)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader_cls = VoxelPoseCampusLoader if args.dataset == "campus" else VoxelPoseShelfLoader

    # Prefer precomputed matched dataset for speed
    precomputed_path = Path(f"outputs/{args.dataset}_matched_dataset.pkl")
    if precomputed_path.exists():
        print(f"Loading precomputed matched dataset from {precomputed_path}...")
        inputs, targets = load_precomputed_dataset(precomputed_path, args.frame_start, args.frame_end)
    else:
        print("Building dataset on the fly...")
        loader = loader_cls(args.data_root)
        inputs, targets = build_dataset(loader, args.frame_start, args.frame_end)
    print(f"Dataset: {inputs.shape[0]} frames, {inputs.shape[1]} views, {inputs.shape[2]} joints")

    # Simple frame-level split
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    best_val = float("inf")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_name = f"attention_fusion_{args.dataset}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            # Combine MSE with MPJPE-style loss for better 3D geometry
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
            torch.save(model.state_dict(), output_dir / output_name)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_MPJPE={val_mpjpe:.4f}"
            )

    print(f"Best val_loss={best_val:.4f}, checkpoint: {output_dir / output_name}")


if __name__ == "__main__":
    main()
