"""CPU-only smoke test: visibility-gated cross-view residual v2 under occlusion.

Generates a tiny synthetic multi-view dataset, trains a small
``CrossviewResidualVisibilityV2`` for a few epochs, and evaluates it under
clean, random view-drop, and random joint-drop conditions.

The script is intentionally lightweight, runs on CPU, and does not touch any
running GPU training jobs.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/eval_visibility_v2_occlusion_robustness_smoke.py
    KMP_DUPLICATE_LIB_OK=TRUE python experiments/eval_visibility_v2_occlusion_robustness_smoke.py --epochs 5 --view_rate 0.3 --joint_rate 0.3
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
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
    X = joints_3d
    t_expanded = t[:, None, None, :]
    X_cam = torch.einsum("vab,fjb->vfja", R, X) + t_expanded
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.matmul(K[:, None, None], (X_cam / z)[..., None])
    points_2d = uv[..., :2, 0] / uv[..., 2:3, 0]
    return points_2d.permute(1, 0, 2, 3)


class SyntheticOcclusionDataset(torch.utils.data.Dataset):
    """Tiny synthetic multi-view pose dataset with optional view/joint occlusion."""

    def __init__(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        n_frames: int = 120,
        n_joints: int = 17,
        clip_len: int = 9,
        noise_std: float = 0.5,
        view_rate: float = 0.0,
        joint_rate: float = 0.0,
        seed: int = 42,
    ):
        self.K = K
        self.R = R
        self.t = t
        self.n_joints = n_joints
        self.clip_len = clip_len
        self.view_rate = view_rate
        self.joint_rate = joint_rate

        g = torch.Generator().manual_seed(seed)
        joints_3d = torch.randn(n_frames, n_joints, 3, generator=g) * 0.3
        for _ in range(2):
            joints_3d[1:-1] = 0.5 * joints_3d[1:-1] + 0.25 * (joints_3d[:-2] + joints_3d[2:])

        points_2d = _project_points(joints_3d, K, R, t)
        if noise_std > 0:
            points_2d = points_2d + torch.randn_like(points_2d) * noise_std

        confidences = torch.ones_like(points_2d[..., 0])

        # View-level occlusion
        if view_rate > 0:
            view_mask = torch.rand(V := points_2d.shape[1], generator=g) < view_rate
            confidences[:, view_mask, :] = 0.0

        # Joint-level occlusion (per view)
        if joint_rate > 0:
            joint_mask = torch.rand(*points_2d.shape[:3], generator=g) < joint_rate
            confidences[joint_mask] = 0.0

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


def evaluate(model, loader, device, visibility_loss_weight: float = 0.1):
    model.eval()
    total_err = 0.0
    total_vis_acc = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _, visibility = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean(dim=(-1, -2))  # per sample
            total_err += err.sum().item()

            visible_target = (xb[..., 2] > 0).float().to(device)
            vis_pred = (visibility > 0.5).float()
            vis_acc = (vis_pred == visible_target).float().mean(dim=(-1, -2, -3))  # per sample
            total_vis_acc += vis_acc.sum().item()
            count += xb.size(0)
    return total_err / count, total_vis_acc / count


def main():
    parser = argparse.ArgumentParser(
        description="CPU smoke: visibility-gated model under synthetic occlusion"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_st_layers", type=int, default=1)
    parser.add_argument("--residual_hidden", type=int, default=64)
    parser.add_argument("--visibility_hidden", type=int, default=32)
    parser.add_argument("--view_rate", type=float, default=0.2, help="Training view occlusion rate")
    parser.add_argument("--joint_rate", type=float, default=0.2, help="Training joint occlusion rate")
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/visibility_v2_occlusion_smoke.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    K, R, t = _make_synthetic_cameras(n_views=4)
    n_joints = 17

    train_dataset = SyntheticOcclusionDataset(
        K, R, t,
        n_frames=120,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_rate=args.view_rate,
        joint_rate=args.joint_rate,
        seed=args.seed,
    )
    val_clean = SyntheticOcclusionDataset(
        K, R, t,
        n_frames=40,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_rate=0.0,
        joint_rate=0.0,
        seed=args.seed + 1,
    )
    val_view = SyntheticOcclusionDataset(
        K, R, t,
        n_frames=40,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_rate=0.3,
        joint_rate=0.0,
        seed=args.seed + 2,
    )
    val_joint = SyntheticOcclusionDataset(
        K, R, t,
        n_frames=40,
        n_joints=n_joints,
        clip_len=args.clip_len,
        view_rate=0.0,
        joint_rate=0.3,
        seed=args.seed + 3,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loaders = {
        "clean": torch.utils.data.DataLoader(
            val_clean, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
        ),
        "view_drop_30": torch.utils.data.DataLoader(
            val_view, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
        ),
        "joint_drop_30": torch.utils.data.DataLoader(
            val_joint, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
        ),
    }

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
    print(f"Model: n_views=4, j={n_joints}, clip_len={args.clip_len}, d={args.d}, params={n_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining with view_rate={args.view_rate}, joint_rate={args.joint_rate}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, Kb, Rb, tb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Kb, Rb, tb = Kb.to(device), Rb.to(device), tb.to(device)
            optimizer.zero_grad()
            pred, _, visibility = model(xb, K=Kb, R=Rb, t=tb)
            loss = criterion(pred, yb)
            visible_target = (xb[..., 2] > 0).float().to(device)
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

    print("\n=== Occlusion robustness (CPU smoke) ===")
    results = {"train_view_rate": args.view_rate, "train_joint_rate": args.joint_rate, "conditions": {}}
    for name, loader in val_loaders.items():
        mpjpe_m, vis_acc = evaluate(model, loader, device, args.visibility_loss_weight)
        mpjpe_mm = mpjpe_m * 1000.0
        results["conditions"][name] = {"mpjpe_mm": mpjpe_mm, "visibility_accuracy": vis_acc}
        print(f"  {name:15s} | MPJPE {mpjpe_mm:7.2f} mm | vis_acc {vis_acc*100:.1f}%")

    torch.save(model.state_dict(), output_path)
    results["checkpoint"] = str(output_path)
    print(f"\nSaved checkpoint to {output_path}")
    return results


if __name__ == "__main__":
    main()
