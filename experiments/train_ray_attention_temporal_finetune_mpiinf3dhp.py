"""Two-stage training for temporal ray-attention fusion on MPI-INF-3DHP.

Stage 1: Pretrain ``RayAttentionFusionModelV4`` on single-frame MPI-INF-3DHP.
Stage 2: Fine-tune ``RayAttentionFusionModelTemporalV4`` initialized from the
stage-1 checkpoint on temporal clips.

Usage
-----
    conda run -n mf python experiments/train_ray_attention_temporal_finetune_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
               data/webbridge/mpi_inf_3dhp/s_01_seq_01_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --stage1_epochs 2 --stage2_epochs 2

The script expects canonical WebBridge .npz files where all spatial data are in
meters (the ``_m`` suffix variants).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.fusion.ray_attention_v4_model import RayAttentionFusionModelV4
from motionflow_mv.fusion.ray_attention_temporal_v4_model import RayAttentionFusionModelTemporalV4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SingleFrameDataset(torch.utils.data.Dataset):
    """Yield single frames (V, J, 3) from a canonical .npz sequence."""

    def __init__(self, npz_path: str, n_samples: int = 4000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()  # (T, J, 3)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        frame_idx = random.randint(0, self.total_frames - 1)
        x = torch.cat(
            [self.points_2d[frame_idx],
             self.confidences[frame_idx].unsqueeze(-1)],
            dim=-1,
        )  # (V, J, 3)
        y = self.joints_3d[frame_idx]  # (J, 3)
        return x, y, self.K, self.R, self.t


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


def augment_frame_or_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1,
                          outlier_rate: float = 0.02, outlier_scale: float = 100.0):
    """Lightweight per-sample / per-clip augmentation."""
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


def evaluate_v4(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def evaluate_temporal(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred, _ = model(xb, K=K, R=R, t=t)
            err = (pred - yb).norm(dim=-1).mean()
            total_err += err.item() * xb.size(0)
            total_count += xb.size(0)
    return total_err / total_count


def stage1_train_v4(args, device, n_views, j):
    """Stage 1: pretrain V4 on single MPI-INF-3DHP frames."""
    print("\n=== Stage 1: pretrain RayAttentionFusionModelV4 on single frames ===")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(SingleFrameDataset(tp, n_samples=args.stage1_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = SingleFrameDataset(args.val, n_samples=args.val_samples)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = RayAttentionFusionModelV4(
        j=j, d=args.d, n_views=n_views, n_heads=4, n_joint_layers=1,
    ).to(device)
    print(f"Stage 1 model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    pretrain_path = Path(args.stage1_checkpoint)
    pretrain_path.parent.mkdir(exist_ok=True, parents=True)

    best_val = float("inf")
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_frame_or_clip(xb.unsqueeze(1)).squeeze(1)  # (B, V, J, 3)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate_v4(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), pretrain_path)
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Stage 1 best val MPJPE: {best_val*1000:.2f}mm -> {pretrain_path}")
    return pretrain_path


def stage2_finetune_temporal(args, device, n_views, j, pretrain_path):
    """Stage 2: fine-tune temporal V4 initialized from the stage-1 checkpoint."""
    print("\n=== Stage 2: fine-tune RayAttentionFusionModelTemporalV4 ===")

    train_datasets = []
    for tp in args.train:
        train_datasets.append(RandomClipDataset(tp, args.clip_len, n_samples=args.stage2_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = RayAttentionFusionModelTemporalV4(
        j=j, d=args.d, n_views=n_views, n_heads=4, n_joint_layers=1,
        n_temporal_layers=args.n_temporal_layers,
    ).to(device)

    # Load V4 weights into the per-frame encoder.
    v4_state = torch.load(pretrain_path, map_location=device)
    model.load_v4_state_dict(v4_state, strict=False)
    print(f"Loaded V4 pretrain checkpoint from {pretrain_path}")
    print(f"Stage 2 model params: {sum(p.numel() for p in model.parameters())}")

    # Optionally freeze the per-frame encoder for the first few epochs.
    if args.freeze_epochs > 0:
        for name, param in model.named_parameters():
            if "temporal" not in name and "temporal_pos_embed" not in name:
                param.requires_grad = False
        print(f"Per-frame encoder frozen for the first {args.freeze_epochs} epochs")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    best_val = float("inf")
    for epoch in range(1, args.stage2_epochs + 1):
        # Unfreeze per-frame encoder after warmup.
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print("Unfroze per-frame encoder")

        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_frame_or_clip(xb)
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate_temporal(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Stage 2 best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Two-stage train V4 -> TemporalV4 on MPI-INF-3DHP")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage1_samples", type=int, default=4000, help="Random single frames per train sequence")
    parser.add_argument("--stage2_samples", type=int, default=2000, help="Random clips per train sequence")
    parser.add_argument("--val_samples", type=int, default=500, help="Random single frames for V4 validation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_epochs", type=int, default=0, help="Freeze per-frame encoder for first N stage-2 epochs")
    parser.add_argument("--stage1_checkpoint", type=str, default="outputs/ray_attention_v4_mpiinf3dhp_pretrain.pth")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip stage 1 and use existing --stage1_checkpoint")
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_v4_mpiinf3dhp.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Infer dimensions from the first train file.
    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}")

    if not args.skip_stage1:
        stage1_train_v4(args, device, n_views, j)
    else:
        print(f"Skipping stage 1, using checkpoint: {args.stage1_checkpoint}")

    stage2_finetune_temporal(args, device, n_views, j, args.stage1_checkpoint)


if __name__ == "__main__":
    main()
