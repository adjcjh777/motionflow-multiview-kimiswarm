"""CPU-only smoke test for the realtime knowledge-distilled student.

Trains a tiny teacher, then distils an even smaller
``DistilledStudentPrincipalPointModel`` on synthetic multi-view data.
Dimensions are deliberately tiny so the whole run completes in seconds on CPU.

Usage
-----
    python experiments/prototypes/iter17_realtime-kd-student_smoke.py

Output
------
    - prints per-epoch train/val metrics
    - saves the student checkpoint to ``outputs/iter17_realtime-kd-student_smoke.pth``
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.models.distilled_student_principal_point_model import (
    DistilledStudentPrincipalPointModel,
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
        n_joints: int = 17,
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
    parser = argparse.ArgumentParser(description="CPU-only smoke test for realtime KD student")
    parser.add_argument("--epochs", type=int, default=2, help="Number of student distillation epochs")
    parser.add_argument("--teacher_epochs", type=int, default=1, help="Number of teacher pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--teacher_d", type=int, default=32, help="Teacher feature dimension")
    parser.add_argument("--student_d", type=int, default=16, help="Student feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="Number of ST transformer layers")
    parser.add_argument("--teacher_residual_hidden", type=int, default=64, help="Teacher residual hidden size")
    parser.add_argument("--student_residual_hidden", type=int, default=32, help="Student residual hidden size")
    parser.add_argument("--distill_alpha", type=float, default=0.5, help="Weight for teacher distillation loss")
    parser.add_argument("--weight_align_beta", type=float, default=0.1, help="Weight for view-weight alignment loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output", type=str, default="outputs/iter17_realtime-kd-student_smoke.pth", help="Output checkpoint path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else "cpu")
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17

    train_dataset = SyntheticSmokeDataset(K, R, t, n_frames=80, n_joints=n_joints, clip_len=args.clip_len)
    val_dataset = SyntheticSmokeDataset(K, R, t, n_frames=30, n_joints=n_joints, clip_len=args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Teacher: small but heavier than the student.
    teacher = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.teacher_d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.teacher_residual_hidden,
        principal_point_hidden=32,
        return_pp_delta=False,
    ).to(device)

    # Student: lightweight real-time model.
    student = DistilledStudentPrincipalPointModel(
        j=n_joints,
        d=args.student_d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.student_residual_hidden,
        principal_point_hidden=32,
        return_pp_delta=False,
    ).to(device)

    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, "
        f"teacher_d={args.teacher_d}, student_d={args.student_d}, "
        f"teacher_params={sum(p.numel() for p in teacher.parameters())}, "
        f"student_params={sum(p.numel() for p in student.parameters())}"
    )

    criterion = nn.MSELoss()

    # Optional lightweight teacher warmup on synthetic data.
    if args.teacher_epochs > 0:
        print(f"Warming up teacher for {args.teacher_epochs} epoch(s)...")
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr)
        for _ in range(args.teacher_epochs):
            teacher.train()
            for xb, yb, Kb, Rb, tb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)
                teacher_optimizer.zero_grad()
                pred, *_ = teacher(xb, K=Kb, R=Rb, t=tb)
                loss = criterion(pred, yb)
                loss.backward()
                teacher_optimizer.step()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # Distil student.
    student_optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        for xb, yb, Kb, Rb, tb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)

            with torch.no_grad():
                teacher_pred, teacher_weights, *_ = teacher(xb, K=Kb, R=Rb, t=tb)

            student_pred, student_weights, *_ = student(xb, K=Kb, R=Rb, t=tb)

            loss = criterion(student_pred, yb)
            if args.distill_alpha > 0.0:
                distill_loss = criterion(student_pred, teacher_pred)
                loss = (1.0 - args.distill_alpha) * loss + args.distill_alpha * distill_loss

            if args.weight_align_beta > 0.0:
                s_w = student_weights.reshape(-1, 4)
                t_w = teacher_weights.reshape(-1, 4)
                mask = (s_w.sum(dim=-1, keepdim=True) > 0) & (t_w.sum(dim=-1, keepdim=True) > 0)
                if mask.any():
                    s_w = s_w[mask.squeeze(-1)]
                    t_w = t_w[mask.squeeze(-1)]
                    cos_sim = F.cosine_similarity(s_w, t_w, dim=-1).mean()
                    loss = loss + args.weight_align_beta * (1.0 - cos_sim)

            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        val_err = evaluate(student, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(student.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"val_MPJPE={val_err*1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
