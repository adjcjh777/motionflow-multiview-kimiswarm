"""Smoke training for Cross-View Graph Attention on a synthetic MPI-INF-3DHP-like dataset.

This script prototypes the ``CrossViewGraphAttention`` module from
``motionflow_mv.fusion.prototypes`` inside a minimal pose-estimation model.
It is intentionally CPU-only, uses tiny dimensions, and runs for at most a
few epochs.

Usage
-----
    python experiments/prototypes/iter17_cross-view-graph-attention_smoke.py
    python experiments/prototypes/iter17_cross-view-graph-attention_smoke.py --epochs 3 --d 32

Output
------
    - prints per-epoch train/val metrics
    - saves the best checkpoint to ``outputs/iter17_cross_view_graph_attention_smoke.pth``
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.fusion.graph_joint_relation import (
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    CrossViewGraphAttention,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthetic_cameras(n_views: int = 4):
    """Build a circular rig of pinhole cameras and return K, R, t tensors."""
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


def _project_points(joints_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Project world points into all views."""
    X = joints_3d  # (F, J, 3)
    t = t[:, None, None, :]  # (V, 1, 1, 3)
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t  # (V, F, J, 3)
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])  # (V, F, J, 3, 1)
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)  # (F, V, J, 2)


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset for smoke tests."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 100,
        n_joints: int = 28,
        clip_len: int = 9,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3  # (F, J, 3)
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d  # (F, V, J, 2)
        self.confidences = torch.ones_like(points_2d[..., 0])  # (F, V, J)
        self.joints_3d = joints_3d  # (F, J, 3)

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


class CrossViewGraphAttentionPoseModel(nn.Module):
    """Minimal pose model using ``CrossViewGraphAttention`` as the core fusion block.

    The model processes per-view, per-joint 2-D observations, applies a stack
    of (view, joint) graph-attention layers, pools across views and time, and
    regresses 3-D joint positions.
    """

    def __init__(
        self,
        n_joints: int = 28,
        n_views: int = 4,
        d: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_views = n_views
        self.d = d

        self.input_proj = nn.Linear(3, d)  # [x, y, confidence] -> d
        self.graph_attn = CrossViewGraphAttention(
            d=d,
            n_views=n_views,
            n_layers=n_layers,
            n_heads=n_heads,
            n_edge_types=4,  # bone, symmetry, cross-view, self
            dropout=0.0,
        )
        self.graph_attn.build_edge_index(
            n_joints, MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V, J, 3)
        Returns:
            (B, T, J, 3)
        """
        B, T, V, J, _ = x.shape
        x = self.input_proj(x)  # (B, T, V, J, d)

        # Process each frame independently.
        x = x.reshape(B * T, V, J, self.d)
        x = self.graph_attn(x)
        x = x.reshape(B, T, V, J, self.d)

        # View pooling.
        x = x.mean(dim=2)  # (B, T, J, d)

        # Temporal pooling.
        x = x.mean(dim=1)  # (B, J, d)

        # Regress 3-D pose and broadcast back to (B, T, J, 3) for loss computation.
        pred = self.output_mlp(x)  # (B, J, 3)
        pred = pred.unsqueeze(1).expand(-1, T, -1, -1)
        return pred


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_err += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
            count += xb.size(0)
    return total_loss / count, total_err / count


def main():
    parser = argparse.ArgumentParser(description="Smoke training for Cross-View Graph Attention")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of graph-attention layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--output", type=str, default="outputs/iter17_cross_view_graph_attention_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    n_views = 4
    n_joints = 28
    K, R, t = _make_synthetic_cameras(n_views=n_views)

    train_dataset = SyntheticSmokeDataset(
        K, R, t,
        n_frames=120,
        n_joints=n_joints,
        clip_len=args.clip_len,
    )
    val_dataset = SyntheticSmokeDataset(
        K, R, t,
        n_frames=40,
        n_joints=n_joints,
        clip_len=args.clip_len,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = CrossViewGraphAttentionPoseModel(
        n_joints=n_joints,
        n_views=n_views,
        d=args.d,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"n_views={n_views}, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_layers={args.n_layers}, n_heads={args.n_heads}, params={n_params}"
    )

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
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_err = evaluate(model, val_loader, device, criterion)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
