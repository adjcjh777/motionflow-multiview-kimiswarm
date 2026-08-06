"""Train the combined visibility + uncertainty cross-view residual v1 model.

The model returns both a learned visibility mask and a per-view log-variance, so
the training objective combines 3-D MSE, BCE visibility loss, and the
reprojection negative-log-likelihood from the uncertainty head.
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

from motionflow_mv.calibration.perturb import perturb_cameras_with_delta
from motionflow_mv.models.crossview_residual_visibility_uncertainty_v1 import (
    CrossviewResidualVisibilityUncertaintyV1,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield clips (T, V, J, 3) from a long canonical .npz sequence."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

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
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


class RandomClipDataset(torch.utils.data.Dataset):
    """Sample random clips from a sequence; useful for train set augmentation."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
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


def augment_clip(
    x,
    noise_std: float = 0.5,
    dropout_rate: float = 0.1,
    outlier_rate: float = 0.02,
    outlier_scale: float = 100.0,
    view_dropout_rate: float = 0.0,
    min_views: int = 2,
):
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
    if view_dropout_rate > 0:
        B = x.shape[0]
        V = x.shape[2]
        view_mask = (torch.rand(B, V, device=x.device) > view_dropout_rate).float()  # 1 = keep
        for i in range(B):
            kept = view_mask[i].nonzero(as_tuple=True)[0]
            if kept.numel() < min_views:
                dropped = (view_mask[i] == 0).nonzero(as_tuple=True)[0]
                needed = min_views - kept.numel()
                if needed > 0 and dropped.numel() > 0:
                    perm = torch.randperm(dropped.numel())
                    extra = dropped[perm[:needed]]
                    view_mask[i, extra] = 1.0
        x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)
    return x


def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _, _, _, nll_loss = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Train visibility + uncertainty cross-view residual v1 on MPI-INF-3DHP"
    )
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--principal_point_hidden", type=int, default=64)
    parser.add_argument("--principal_point_max_offset", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pp_loss_weight", type=float, default=0.05)
    parser.add_argument("--focal_max_scale", type=float, default=0.0)
    parser.add_argument("--cam_aug_rot", type=float, default=0.5)
    parser.add_argument("--cam_aug_trans", type=float, default=0.005)
    parser.add_argument("--cam_aug_focal", type=float, default=0.01)
    parser.add_argument("--cam_aug_pp", type=float, default=5.0)
    parser.add_argument("--cam_aug_schedule", type=str, default="flat", choices=["flat", "extrinsic_curriculum"])
    parser.add_argument("--cam_aug_ramp_epochs", type=int, default=10)
    parser.add_argument("--view_dropout_rate", type=float, default=0.2)
    parser.add_argument("--min_views", type=int, default=4)
    parser.add_argument("--visibility_loss_weight", type=float, default=0.1)
    parser.add_argument("--uncertainty_loss_weight", type=float, default=0.1)
    parser.add_argument("--log_var_min", type=float, default=-10.0)
    parser.add_argument("--log_var_max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm_start", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ray_attention_temporal_crossview_residual_visibility_uncertainty_v1_mpiinf3dhp.pth",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=args.val_stride)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(
        f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, "
        f"n_st_layers={args.n_st_layers}, residual_hidden={args.residual_hidden}, "
        f"visibility_loss_weight={args.visibility_loss_weight}, "
        f"uncertainty_loss_weight={args.uncertainty_loss_weight}"
    )

    model = CrossviewResidualVisibilityUncertaintyV1(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        principal_point_hidden=args.principal_point_hidden,
        principal_point_max_offset=args.principal_point_max_offset,
        focal_max_scale=args.focal_max_scale,
        visibility_loss_weight=args.visibility_loss_weight,
        uncertainty_loss_weight=args.uncertainty_loss_weight,
        log_var_min=args.log_var_min,
        log_var_max=args.log_var_max,
    ).to(device)

    if args.warm_start is not None:
        state = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys when warm-starting: {missing[:5]}")
        if unexpected:
            print(f"Warning: unexpected keys when warm-starting (ignored): {unexpected[:5]}")
        print(f"Warm-started from {args.warm_start}")

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        if args.cam_aug_schedule == "extrinsic_curriculum":
            ramp = min(1.0, epoch / max(1, args.cam_aug_ramp_epochs))
            schedule_rot = args.cam_aug_rot * ramp
            schedule_trans = args.cam_aug_trans * ramp
        else:
            schedule_rot = args.cam_aug_rot
            schedule_trans = args.cam_aug_trans
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb, view_dropout_rate=args.view_dropout_rate, min_views=args.min_views)
            K, R, t, true_pp_delta, true_focal_scale = perturb_cameras_with_delta(
                K,
                R,
                t,
                rot_std=schedule_rot,
                trans_std=schedule_trans,
                focal_std=args.cam_aug_focal,
                pp_std=args.cam_aug_pp,
            )
            optimizer.zero_grad()
            pred, _, visibility, log_var, nll_loss = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)

            # Visibility BCE: target is 1 for views with positive detector confidence.
            visible_target = (xb[..., 2] > 0).float().to(device)
            vis_loss = F.binary_cross_entropy(visibility, visible_target)
            loss = loss + args.visibility_loss_weight * vis_loss + nll_loss

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
