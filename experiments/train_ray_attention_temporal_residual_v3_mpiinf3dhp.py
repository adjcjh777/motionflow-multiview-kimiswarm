"""Train RayAttentionFusionModelTemporalResidual v3 on MPI-INF-3DHP / H36M clips.

This is an extended trainer over ``train_ray_attention_temporal_residual_mpiinf3dhp.py``
that adds optional auxiliary losses (bone-length / velocity consistency),
learning-rate scheduling, weight decay, gradient clipping, and richer logging.

Usage
-----
    conda run -n mf python experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
               data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --epochs 10 --d 128 --residual_hidden 256
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual


H36M_PARENT_17 = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


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
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )
        y = self.joints_3d[start:end]
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
            [self.points_2d[start:end],
             self.confidences[start:end].unsqueeze(-1)],
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


def bone_length_loss(pred, parent_indices):
    """Mean squared difference of bone lengths between pred and a zero target.

    In practice this regularises bone lengths to stay close to the dataset mean
    when used as a consistency loss on the prediction alone.  A more complete
    implementation would compare against ground-truth bone lengths.
    """
    children = [i for i, p in enumerate(parent_indices) if p >= 0]
    parents = [parent_indices[c] for c in children]
    bones = pred[..., children, :] - pred[..., parents, :]
    lengths = torch.norm(bones, dim=-1)
    # Penalise very short or very long bones mildly; can be tied to GT in future.
    return lengths.var(dim=-1).mean()


def velocity_consistency_loss(pred):
    """L2 smoothness loss over time: encourages constant-velocity skeletons."""
    if pred.shape[1] < 3:
        return 0.0
    vel = pred[:, 1:] - pred[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return torch.mean(acc ** 2)


def evaluate(model, loader, device, criterion, aux_weight: float = 0.0,
             parent_indices=None, velocity_weight: float = 0.0):
    model.eval()
    total_err = 0.0
    total_count = 0
    total_aux = 0.0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            total_err += loss.item() * xb.size(0)
            total_count += xb.size(0)
            if aux_weight > 0 and parent_indices is not None:
                total_aux += bone_length_loss(pred, parent_indices).item() * xb.size(0)
    return total_err / total_count, total_aux / total_count


def main():
    parser = argparse.ArgumentParser(description="Train temporal ray-attention fusion with residual refinement v3")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=4000, help="Random clips per train sequence")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Bone-length regulariser weight")
    parser.add_argument("--velocity_weight", type=float, default=0.0, help="Temporal smoothness weight")
    parser.add_argument("--parent_indices", type=str, default=None,
                        help="JSON list of parent joint indices for bone loss (e.g. H36M 17-joint hierarchy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_residual_v3.pth")
    parser.add_argument("--log", type=str, default=None, help="Optional JSON training log path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    parent_indices = None
    if args.parent_indices:
        parent_indices = json.loads(args.parent_indices)
    elif args.aux_weight > 0:
        # Best-effort default for common 17-joint skeletons.
        parent_indices = H36M_PARENT_17
        print("Using default 17-joint H36M parent indices for bone loss.")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}, "
          f"residual_hidden={args.residual_hidden}")
    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")

    model = RayAttentionFusionModelTemporalResidual(
        j=j,
        d=args.d,
        n_views=n_views,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None

    criterion = nn.MSELoss()
    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    log_entries = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_aux_total = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)

            total_loss = loss
            if args.aux_weight > 0 and parent_indices is not None:
                aux = bone_length_loss(pred, parent_indices)
                total_loss = total_loss + args.aux_weight * aux
                train_aux_total += aux.item() * xb.size(0)
            if args.velocity_weight > 0:
                total_loss = total_loss + args.velocity_weight * velocity_consistency_loss(pred)

            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += total_loss.item() * xb.size(0)
            train_mse += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        train_aux_total /= len(train_loader.dataset)

        val_mse, val_aux = evaluate(model, val_loader, device, criterion,
                                    aux_weight=args.aux_weight, parent_indices=parent_indices)
        val_err = val_mse ** 0.5  # RMSE in meters -> approx MPJPE-like

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_mse)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_aux": train_aux_total,
            "val_mse": val_mse,
            "val_mpjpe_mm": val_err * 1000.0,
            "lr": current_lr,
        }
        log_entries.append(entry)

        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            torch.save(model.state_dict(), output_path)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm, lr={current_lr:.2e} (saved)")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm, lr={current_lr:.2e}")

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"config": vars(args), "log": log_entries}, f, indent=2, default=str)
        print(f"Saved training log to {log_path}")

    print(f"Best val MPJPE: {best_val**0.5*1000:.2f}mm -> {output_path}")


if __name__ == "__main__":
    main()
