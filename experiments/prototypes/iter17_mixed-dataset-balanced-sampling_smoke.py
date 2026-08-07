"""Smoke test for mixed-dataset balanced sampling.

This prototype adds :class:`BalancedMixedSampler`, which samples uniformly across
sub-datasets of a ``ConcatDataset`` instead of letting the largest dataset
dominate training.  The script runs a tiny CPU-only training loop on three
synthetic multi-view sequences of very different lengths and verifies that each
"dataset" is represented roughly equally in training batches.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.data.webbridge_mixed_dataset import BalancedMixedSampler
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _project_points(joints_3d: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    X = joints_3d
    t = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None, :], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


class SyntheticDataset(torch.utils.data.Dataset):
    """One synthetic multi-view sequence tagged with a dataset id."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        dataset_id: int,
        n_frames: int,
        n_joints: int,
        clip_len: int,
        noise_std: float = 0.5,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.dataset_id = dataset_id
        self.clip_len = clip_len
        self.noise_std = noise_std

        torch.manual_seed(42 + dataset_id)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, n_frames - clip_len + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t, self.dataset_id


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, dataset_ids


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    for xb, yb, K, R, t, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        pred, *_ = model(xb, K=K, R=R, t=t)
        err = (pred - yb).norm(dim=-1).mean()
        total_err += err.item() * xb.size(0)
        total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(description="Mixed-dataset balanced sampling smoke")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="ST transformer layers")
    parser.add_argument("--residual_hidden", type=int, default=64, help="Residual MLP hidden size")
    parser.add_argument("--train_samples_per_epoch", type=int, default=32, help="Samples per epoch with balanced sampler")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output", type=str, default="outputs/iter17_mixed_dataset_balanced_sampling_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17

    # Three synthetic "datasets" with very different frame counts.
    # Without balanced sampling the largest one (ds=2) would dominate.
    dataset_configs = [
        {"id": 0, "n_frames": 60},
        {"id": 1, "n_frames": 150},
        {"id": 2, "n_frames": 300},
    ]

    train_datasets = []
    for cfg in dataset_configs:
        train_datasets.append(
            SyntheticDataset(
                K, R, t,
                dataset_id=cfg["id"],
                n_frames=cfg["n_frames"],
                n_joints=n_joints,
                clip_len=args.clip_len,
            )
        )

    val_dataset = SyntheticDataset(
        K, R, t,
        dataset_id=0,
        n_frames=40,
        n_joints=n_joints,
        clip_len=args.clip_len,
    )

    train_concat = torch.utils.data.ConcatDataset(train_datasets)
    train_sampler = BalancedMixedSampler(train_concat, num_samples=args.train_samples_per_epoch)

    train_loader = torch.utils.data.DataLoader(
        train_concat,
        batch_size=args.batch_size,
        sampler=train_sampler,
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

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=32,
        principal_point_max_offset=5.0,
        focal_max_scale=0.0,
        return_pp_delta=False,
    ).to(device)

    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
        f"params={sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"Train sub-dataset sizes: {[len(ds) for ds in train_datasets]} "
        f"(balanced sampler draws {args.train_samples_per_epoch} per epoch)"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        dataset_counts = Counter()

        for xb, yb, Kb, Rb, tb, ids in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)

            optimizer.zero_grad()
            pred, *_ = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_count += xb.size(0)
            for i in ids.tolist():
                dataset_counts[i] += 1

        train_loss /= train_count
        val_err = evaluate(model, val_loader, device)

        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            status = "saved"
        else:
            status = ""

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_MPJPE={val_err*1000:.2f}mm "
            f"batch_dataset_counts={dict(sorted(dataset_counts.items()))} "
            f"{status}"
        )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
