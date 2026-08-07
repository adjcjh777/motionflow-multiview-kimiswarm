"""Iter-17 smoke test: semi-supervised teacher-student pseudo-labeling.

This prototype demonstrates the semi-supervised pipeline on a tiny synthetic
circular camera rig.  It is CPU-only, uses tiny dimensions (d<=32, batch<=2,
clip_len<=9), and runs at most 3 epochs.

Teacher-student flow
--------------------
1. Train a teacher on a small labeled split (1 smoke epoch).
2. Generate confidence-weighted 3-D pseudo-labels for an unlabeled split.
3. Train a student on alternating labeled + pseudo-labeled mini-batches.

Usage
-----
    python experiments/prototypes/iter17_semi-supervised-pseudo-labeling_smoke.py
    python experiments/prototypes/iter17_semi-supervised-pseudo-labeling_smoke.py --epochs 2 --teacher_epochs 1

Output
------
    - prints per-epoch train/val metrics
    - saves a checkpoint to ``outputs/iter17_semi_supervised_pseudo_labeling_smoke.pth``
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


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                 outlier_rate: float = 0.02, outlier_scale: float = 100.0):
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) > dropout_rate).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device) < outlier_rate
        outlier = (torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device) - 0.5) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def _reprojection_error(pred_3d, points_2d, K, R, t, confidences, eps=1e-6):
    """Per-sample reprojection MSE (B,) for a batch of clips."""
    B, T, J, _ = pred_3d.shape
    K = K.unsqueeze(1)  # (B, 1, V, 3, 3)
    R = R.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3, 3)
    t = t.unsqueeze(1).unsqueeze(-2)  # (B, 1, V, 1, 3)
    X = pred_3d.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, J, 3, 1)
    X_cam = (R @ X).squeeze(-1) + t  # (B, T, V, J, 3)
    z = X_cam[..., 2:3]
    proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)
    proj_2d = proj[..., :2] / (z.clamp(min=eps))
    diff = proj_2d - points_2d
    sq = (diff ** 2).sum(dim=-1)  # (B, T, V, J)
    weight = confidences
    per_sample = (sq * weight).sum(dim=(-1, -2, -3)) / weight.sum(dim=(-1, -2, -3)).clamp(min=eps)
    return per_sample


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
        labeled: bool = True,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.noise_std = noise_std
        self.labeled = labeled

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
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        if self.labeled:
            y = self.joints_3d[start:end]
            return x, y, self.K, self.R, self.t
        return x, self.K, self.R, self.t


def collate_labeled(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def collate_unlabeled(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    K = torch.stack([b[1] for b in batch], dim=0)
    R = torch.stack([b[2] for b in batch], dim=0)
    t = torch.stack([b[3] for b in batch], dim=0)
    return x, K, R, t


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb, K, R, t = batch[:5]
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, *_ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


@torch.no_grad()
def generate_pseudo_labels(model, loader, device, conf_thresh: float = 5.0):
    """Generate pseudo-labels and confidence weights for unlabeled clips."""
    model.eval()
    results = []
    for batch in loader:
        xb, K, R, t = batch
        xb = xb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        pred, *_ = model(xb, K=K, R=R, t=t)
        points_2d = xb[..., :2]
        conf = xb[..., 2]
        reproj_per_sample = _reprojection_error(pred, points_2d, K, R, t, conf)
        weight = torch.sigmoid((conf_thresh - reproj_per_sample) / conf_thresh)
        for i in range(xb.size(0)):
            results.append((xb[i].cpu(), pred[i].cpu(), K[i].cpu(), R[i].cpu(), t[i].cpu(), weight[i].cpu()))
    return results


def train_teacher(model, loader, val_loader, args, device):
    """Train teacher on labeled data for a few smoke epochs."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            optimizer.zero_grad()
            pred, *_ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(loader.dataset)
        val_err = evaluate(model, val_loader, device)
        print(f"  Teacher epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")


def main():
    parser = argparse.ArgumentParser(
        description="Iter-17 CPU smoke test for semi-supervised pseudo-labeling"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Student training epochs")
    parser.add_argument("--teacher_epochs", type=int, default=1, help="Teacher training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="Spatio-temporal transformer layers")
    parser.add_argument("--residual_hidden", type=int, default=64, help="Residual MLP hidden size")
    parser.add_argument("--lambda_pseudo", type=float, default=0.5, help="Weight for pseudo-label loss")
    parser.add_argument("--pseudo_conf_thresh", type=float, default=5.0, help="Reprojection error threshold (pixels)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--output", type=str, default="outputs/iter17_semi_supervised_pseudo_labeling_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17

    # Splits: labeled (small), unlabeled (no joints_3d), validation (no overlap)
    labeled_dataset = SyntheticSmokeDataset(
        K, R, t, n_frames=120, n_joints=n_joints, clip_len=args.clip_len, labeled=True,
    )
    unlabeled_dataset = SyntheticSmokeDataset(
        K, R, t, n_frames=120, n_joints=n_joints, clip_len=args.clip_len, labeled=False,
    )
    val_dataset = SyntheticSmokeDataset(
        K, R, t, n_frames=40, n_joints=n_joints, clip_len=args.clip_len, labeled=True,
    )

    labeled_loader = torch.utils.data.DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_labeled, num_workers=0,
    )
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_unlabeled, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_labeled, num_workers=0,
    )

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=32,
        principal_point_max_offset=20.0,
        focal_max_scale=0.0,
        return_pp_delta=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
        f"params={n_params}"
    )

    # ------------------------------------------------------------------
    # Teacher training
    # ------------------------------------------------------------------
    teacher = model
    print(f"Training teacher for {args.teacher_epochs} epoch(s)...")
    train_teacher(teacher, labeled_loader, val_loader, args, device)

    # ------------------------------------------------------------------
    # Generate pseudo-labels
    # ------------------------------------------------------------------
    print("Generating pseudo-labels for unlabeled data...")
    pseudo_items = generate_pseudo_labels(teacher, unlabeled_loader, device, conf_thresh=args.pseudo_conf_thresh)
    print(f"Generated {len(pseudo_items)} pseudo-labeled clips")

    pseudo_loader = torch.utils.data.DataLoader(
        pseudo_items,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            torch.stack([b[1] for b in batch], dim=0),
            torch.stack([b[2] for b in batch], dim=0),
            torch.stack([b[3] for b in batch], dim=0),
            torch.stack([b[4] for b in batch], dim=0),
            torch.stack([b[5] for b in batch], dim=0),
        ),
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Student training
    # ------------------------------------------------------------------
    student = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=32,
        principal_point_max_offset=20.0,
        focal_max_scale=0.0,
        return_pp_delta=True,
    ).to(device)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    pseudo_iter = iter(pseudo_loader)
    labeled_iter = iter(labeled_loader)

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        n_steps = 0

        for _ in range(len(labeled_loader)):
            try:
                x_l, y_l, K_l, R_l, t_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_l, y_l, K_l, R_l, t_l = next(labeled_iter)

            try:
                x_p, y_p, K_p, R_p, t_p, w_p = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                x_p, y_p, K_p, R_p, t_p, w_p = next(pseudo_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            K_l, R_l, t_l = K_l.to(device), R_l.to(device), t_l.to(device)
            x_p, y_p = x_p.to(device), y_p.to(device)
            K_p, R_p, t_p = K_p.to(device), R_p.to(device), t_p.to(device)
            w_p = w_p.to(device)

            x_l = augment_clip(x_l, outlier_rate=0.0)
            x_p = augment_clip(x_p, outlier_rate=0.0)

            optimizer.zero_grad()

            pred_l, *_ = student(x_l, K=K_l, R=R_l, t=t_l)
            loss_l = criterion(pred_l, y_l)

            pred_p, *_ = student(x_p, K=K_p, R=R_p, t=t_p)
            loss_p = (w_p.view(-1, 1, 1, 1) * (pred_p - y_p).pow(2)).mean()

            loss = loss_l + args.lambda_pseudo * loss_p
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * (x_l.size(0) + x_p.size(0))
            n_steps += x_l.size(0) + x_p.size(0)

        train_loss /= max(n_steps, 1)
        val_err = evaluate(student, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(student.state_dict(), output_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm"
            )

    print(f"Best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
