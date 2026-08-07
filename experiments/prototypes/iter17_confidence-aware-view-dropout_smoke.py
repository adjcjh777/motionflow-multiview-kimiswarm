"""Iter-17 smoke: confidence-aware view dropout for multi-view pose."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.data.confidence_resample_dropout import (
    confidence_resample_view_dropout,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_synthetic_cameras(n_views: int = 4):
    K_list, R_list, t_list = [], [], []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)
    K = torch.from_numpy(np.stack(K_list, axis=0)).float()
    R = torch.from_numpy(np.stack(R_list, axis=0)).float()
    t = torch.from_numpy(np.stack(t_list, axis=0)).float()
    return K, R, t


def _project_points(joints_3d, K, R, t):
    X = joints_3d
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    def __init__(self, K, R, t, n_frames=100, n_joints=17, clip_len=9,
                 noise_std=0.5, confidence_view_bias=True):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        confidences = torch.ones_like(points_2d[..., 0])
        if confidence_view_bias:
            confidences[:, 0, :] *= 2.0
            confidences[:, 1, :] *= 0.25
        else:
            confidences = torch.rand_like(confidences) * 1.5 + 0.5

        self.points_2d = points_2d
        self.confidences = confidences
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - clip_len + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx
        end = start + self.clip_len
        x = torch.cat([self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)], dim=-1)
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred = model(xb, K=K, R=R, t=t)[0]
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_err += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
            count += xb.size(0)
    return total_loss / count, total_err / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_st_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/iter17_confidence-aware-view-dropout_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17
    n_train_frames = 120
    n_val_frames = 40

    train_dataset = SyntheticSmokeDataset(K, R, t, n_frames=n_train_frames, n_joints=n_joints, clip_len=args.clip_len)
    val_dataset = SyntheticSmokeDataset(K, R, t, n_frames=n_val_frames, n_joints=n_joints, clip_len=args.clip_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints, d=args.d, n_views=4, n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden, return_pp_delta=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
          f"dropout_rate={args.dropout_rate}, min_views={args.min_views}, "
          f"resample={args.resample}, params={n_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, Kb, Rb, tb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)
            if args.dropout_rate > 0.0:
                xb = confidence_resample_view_dropout(
                    xb, dropout_rate=args.dropout_rate, resample=args.resample, min_views=args.min_views,
                )
            optimizer.zero_grad()
            pred = model(xb, K=Kb, R=Rb, t=tb)[0]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_err = evaluate(model, val_loader, device, criterion)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
