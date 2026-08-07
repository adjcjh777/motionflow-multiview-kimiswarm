"""Demo: train a tiny Bayesian-triangulation model with trainer_v2.

This prototype shows how ``motionflow_mv.training.TrainerV2`` /
``MultiViewPoseTrainerV2`` can replace the ad-hoc training loop in
``experiments/train_bayesian_tri_v2_smoke.py`` while gaining cosine LR with
warmup, gradient clipping, AMP, and EMA.

Run
----
    python experiments/prototypes/swarm_iter18/demo_trainer_v2_on_bayesian_tri_smoke.py

Output
-------
    Prints per-epoch train/val MPJPE and saves the best checkpoint to
    ``outputs/swarm_iter18/demo_trainer_v2.pth``.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)
from motionflow_mv.training import MultiViewPoseTrainerV2


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


class SyntheticSmokeDataset(torch.utils.data.Dataset):
    def __init__(self, K, R, t, n_frames=100, n_joints=17, clip_len=9, noise_std=0.5):
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

        points_2d = self._project_points(joints_3d)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        self.points_2d = points_2d
        self.confidences = torch.ones_like(points_2d[..., 0])
        self.joints_3d = joints_3d
        self.total_frames = n_frames
        self.num_clips = max(1, self.total_frames - clip_len + 1)

    def _project_points(self, joints_3d):
        X = joints_3d  # (F, J, 3)
        t = self.t[:, None, None, :]
        X_cam = torch.einsum("vab,fjb->vfja", self.R, X) + t
        z = X_cam[..., 2:3].clamp(min=1e-6)
        uv = torch.matmul(self.K[:, None, None], (X_cam / z)[..., None])
        points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
        return points_2d.permute(1, 0, 2, 3)

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
        return x, y, self.K, self.R, self.t


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/swarm_iter18/demo_trainer_v2.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17

    train_dataset = SyntheticSmokeDataset(K, R, t, n_frames=120, n_joints=n_joints, clip_len=args.clip_len)
    val_dataset = SyntheticSmokeDataset(K, R, t, n_frames=40, n_joints=n_joints, clip_len=args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = RayAttentionFusionModelBayesianTriV2(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=1,
        residual_hidden=64,
        gn_iters=2,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = MultiViewPoseTrainerV2(
        model,
        optimizer,
        device,
        total_epochs=args.epochs,
        warmup_epochs=max(1, args.epochs // 4),
        eta_min=1e-6,
        max_grad_norm=1.0,
        amp_enabled=True,
        ema_decay=0.999,
        ema_eval=True,
    )

    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_path=args.output,
        save_best=True,
    )

    for entry in history:
        print(
            f"Epoch {entry['epoch']}: "
            f"train_loss={entry['train']['loss']:.6f}, "
            f"val_loss={entry['val']['loss']:.6f}, "
            f"val_MPJPE={entry['val']['mpjpe'] * 1000:.2f}mm"
        )
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
