"""Standalone smoke trainer for the physics-informed skeleton dynamics prior.

This trainer is intentionally lightweight and self-contained: it generates a
small synthetic multi-view sequence, instantiates the physics-augmented
variant of the iter14 anchor model, and runs a few training epochs.  It is
meant as a fast way to verify that the new model and loss compose correctly
before committing to a full MPI-INF-3DHP / H36M / WebBridge run.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.synthetic_3d_dataset import generate_sequence
from motionflow_mv.fusion.graph_joint_relation import H36M_17_PARENTS
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_physics_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics,
)
from motionflow_mv.losses.physics_informed_dynamics import PhysicsInformedSkeletonDynamicsLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_synthetic_npz(output_path: str, n_frames: int = 200, n_views: int = 4, j: int = 17):
    inputs, _, gt, cameras = generate_sequence(
        n_frames=n_frames, n_views=n_views, j=j, noise_std=1.0
    )
    points_2d = inputs[..., :2]
    confidences = inputs[..., 2]
    K = np.stack([cam.K for cam in cameras], axis=0)
    R = np.stack([cam.R for cam in cameras], axis=0)
    t = np.stack([cam.t for cam in cameras], axis=0)
    np.savez_compressed(
        output_path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=gt,
        camera_K=K,
        camera_R=R,
        camera_t=t,
    )


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - self.clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Smoke trainer for physics-informed skeleton dynamics prior")
    parser.add_argument("--data", type=str, default=None, help="Path to a canonical .npz file (if None, a synthetic one is generated)")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--dynamics_hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--val_stride", type=int, default=20)
    parser.add_argument("--physics_loss_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/physics_dynamics_pp_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data is None:
        data_path = "tmp/physics_smoke_synthetic.npz"
        Path(data_path).parent.mkdir(exist_ok=True, parents=True)
        build_synthetic_npz(data_path)
    else:
        data_path = args.data

    train_dataset = TemporalClipDataset(data_path, args.clip_len, stride=1)
    val_dataset = TemporalClipDataset(data_path, args.clip_len, stride=args.val_stride)

    # Subsample random clips for a fast smoke run.
    train_indices = torch.randperm(len(train_dataset))[: args.train_samples].tolist()
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    sample = np.load(data_path)
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, "
          f"residual_hidden={args.residual_hidden}, n_st_layers={args.n_st_layers}, "
          f"dynamics_hidden={args.dynamics_hidden}")

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointPhysics(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        dynamics_hidden=args.dynamics_hidden,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    physics_loss_fn = PhysicsInformedSkeletonDynamicsLoss(
        parents=H36M_17_PARENTS,
        foot_indices=[3, 6, 10, 13],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            if args.physics_loss_weight > 0.0:
                loss = loss + args.physics_loss_weight * physics_loss_fn(pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
