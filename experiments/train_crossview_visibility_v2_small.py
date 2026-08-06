"""CPU smoke training for the visibility-gated cross-view residual v2 model.

Runs a minimal training loop for ``CrossviewResidualVisibilityV2`` on a tiny
synthetic multi-view dataset.  The goal is to verify that the model can be
instantiated, its cross-view visibility head can be differentiated, and a few
epochs of training complete without errors on CPU.

Usage
-----
    python experiments/train_crossview_visibility_v2_small.py
    python experiments/train_crossview_visibility_v2_small.py --epochs 5 --occlusion_rate 0.3
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.models.crossview_residual_visibility_v2 import (
    CrossviewResidualVisibilityV2,
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
    """Project world points into all views.

    Args:
        joints_3d: (F, J, 3)
        K, R, t: (V, 3, 3), (V, 3, 3), (V, 3)

    Returns:
        points_2d: (F, V, J, 2)
    """
    X = joints_3d
    t_expanded = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t_expanded
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


class SyntheticVisibilityDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset with optional view occlusion."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 120,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
        occlusion_rate: float = 0.0,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.occlusion_rate = occlusion_rate

        torch.manual_seed(42)
        joints_3d = torch.randn(n_frames, n_joints, 3) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        # Randomly drop views to simulate occlusion.
        confidences = torch.ones_like(points_2d[..., 0])
        if occlusion_rate > 0:
            mask = torch.rand_like(confidences) > occlusion_rate
            confidences = confidences * mask.float()

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


def evaluate(model, loader, device, criterion, visibility_loss_weight: float = 0.1):
    model.eval()
    total_loss = 0.0
    total_err = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _, visibility = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            visible_target = (xb[..., 2] > 0).float().to(device)
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + visibility_loss_weight * vis_loss
            total_loss += loss.item() * xb.size(0)
            total_err += (pred - yb).norm(dim=-1).mean().item() * xb.size(0)
            count += xb.size(0)
    return total_loss / count, total_err / count


def main():
    parser = argparse.ArgumentParser(description="CPU smoke training for CrossviewResidualVisibilityV2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--clip_len", type=int, default=9, help="Temporal clip length")
    parser.add_argument("--d", type=int, default=32, help="Feature dimension")
    parser.add_argument("--n_st_layers", type=int, default=1, help="Spatio-temporal transformer layers")
    parser.add_argument("--residual_hidden", type=int, default=64, help="Residual MLP hidden size")
    parser.add_argument("--visibility_hidden", type=int, default=32, help="Visibility MLP hidden size")
    parser.add_argument("--occlusion_rate", type=float, default=0.0, help="Synthetic occlusion rate")
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1, help="Weight for BCE visibility loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--output", type=str, default="outputs/crossview_residual_visibility_v2_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17
    n_train_frames = 120
    n_val_frames = 40

    train_dataset = SyntheticVisibilityDataset(
        K, R, t,
        n_frames=n_train_frames,
        n_joints=n_joints,
        clip_len=args.clip_len,
        occlusion_rate=args.occlusion_rate,
    )
    val_dataset = SyntheticVisibilityDataset(
        K, R, t,
        n_frames=n_val_frames,
        n_joints=n_joints,
        clip_len=args.clip_len,
        occlusion_rate=args.occlusion_rate,
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

    model = CrossviewResidualVisibilityV2(
        j=n_joints,
        d=args.d,
        n_views=4,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        visibility_hidden=args.visibility_hidden,
        principal_point_max_offset=0.0,
        focal_max_scale=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, params={n_params}"
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
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)
            optimizer.zero_grad()
            pred, _, visibility = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)

            # Visibility BCE loss: target is 1 for views with positive detector confidence.
            visible_target = (xb[..., 2] > 0).float().to(device)
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_err = evaluate(model, val_loader, device, criterion, args.visibility_loss_weight)
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
