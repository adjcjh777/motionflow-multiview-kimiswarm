"""Two-stage training for the residual temporal ray-attention model.

Stage 1
-------
Pre-train ``RayAttentionFusionModelV4`` on *single-frame* MPI-INF-3DHP.  This
learns a robust per-frame encoder before any temporal modelling is introduced.

Stage 2
-------
Load the V4 checkpoint into the per-frame encoder of
``RayAttentionFusionModelTemporalResidualV2`` (which uses the same normalised
camera embedding as V4) and fine-tune the full model on video clips.

Usage
-----
    conda run -n mf python experiments/train_ray_attention_temporal_residual_v2_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --stage1_epochs 10 --stage2_epochs 30

If a stage-1 checkpoint already exists, pass it with ``--stage1_checkpoint`` to
skip pre-training.
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
from motionflow_mv.fusion.ray_attention_temporal_residual_v2_model import (
    RayAttentionFusionModelTemporalResidualV2,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
class RandomFrameDataset(torch.utils.data.Dataset):
    """Sample single random frames from one or more .npz sequences."""

    def __init__(self, npz_paths, n_samples: int = 4000):
        self.npz_paths = [Path(p) for p in npz_paths]
        self.sequences = []
        self.n_samples = n_samples
        for p in self.npz_paths:
            data = np.load(p)
            self.sequences.append({
                "points_2d": torch.from_numpy(data["points_2d"]).float(),
                "confidences": torch.from_numpy(data["confidences"]).float(),
                "joints_3d": torch.from_numpy(data["joints_3d"]).float(),
                "K": torch.from_numpy(data["camera_K"]).float(),
                "R": torch.from_numpy(data["camera_R"]).float(),
                "t": torch.from_numpy(data["camera_t"]).float(),
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_idx = random.randint(0, len(self.sequences) - 1)
        seq = self.sequences[seq_idx]
        frame_idx = random.randint(0, seq["points_2d"].shape[0] - 1)
        x = torch.cat(
            [seq["points_2d"][frame_idx], seq["confidences"][frame_idx].unsqueeze(-1)],
            dim=-1,
        )  # (V, J, 3)
        y = seq["joints_3d"][frame_idx]  # (J, 3)
        return x, y, seq["K"], seq["R"], seq["t"]


class SequentialFrameDataset(torch.utils.data.Dataset):
    """Yield every frame from a .npz as a single-frame sample."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.joints_3d = torch.from_numpy(data["joints_3d"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.n_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        x = torch.cat(
            [self.points_2d[idx], self.confidences[idx].unsqueeze(-1)],
            dim=-1,
        )
        return x, self.joints_3d[idx], self.K, self.R, self.t


class TemporalClipDataset(torch.utils.data.Dataset):
    """Yield fixed-length clips from a .npz."""

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
    """Sample random clips from a .npz."""

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


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate_single_frame(model, loader, device):
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


# --------------------------------------------------------------------------- #
# Stage 1: pre-train V4 on single frames
# --------------------------------------------------------------------------- #
def run_stage1(args, device, n_views, j):
    print("\n===== Stage 1: pre-training RayAttentionFusionModelV4 (single-frame) =====")

    train_dataset = RandomFrameDataset(args.train, n_samples=args.stage1_train_samples)
    val_dataset = SequentialFrameDataset(args.val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = RayAttentionFusionModelV4(j=j, d=args.d, n_views=n_views).to(device)
    print(f"V4 params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    output_path = Path(args.stage1_output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    best_val = float("inf")

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb.unsqueeze(1)).squeeze(1)  # (B, V, J, 3) augmentation
            optimizer.zero_grad()
            pred, _ = model(xb, K=K, R=R, t=t)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        val_err = evaluate_single_frame(model, val_loader, device)
        if val_err < best_val:
            best_val = val_err
            torch.save(model.state_dict(), output_path)
            print(f"Stage1 Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Stage1 Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Stage1 best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# Stage 2: fine-tune residual temporal model initialized from V4
# --------------------------------------------------------------------------- #
def run_stage2(args, device, n_views, j, stage1_checkpoint):
    print("\n===== Stage 2: fine-tuning RayAttentionFusionModelTemporalResidualV2 =====")

    train_datasets = [RandomClipDataset(tp, args.clip_len, n_samples=args.stage2_train_samples) for tp in args.train]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = TemporalClipDataset(args.val, args.clip_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = RayAttentionFusionModelTemporalResidualV2(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    print(f"TemporalResidualV2 params: {sum(p.numel() for p in model.parameters())}")

    # Load V4 per-frame encoder into the temporal model.
    if stage1_checkpoint is not None and Path(stage1_checkpoint).exists():
        print(f"Loading V4 checkpoint: {stage1_checkpoint}")
        state = torch.load(stage1_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded V4 weights: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys (expected temporal/residual only): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print("WARNING: no V4 checkpoint provided/found; training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    best_val = float("inf")

    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, K, R, t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            xb = augment_clip(xb)
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
            print(f"Stage2 Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm (saved)")
        else:
            print(f"Stage2 Epoch {epoch}: train_loss={train_loss:.6f}, val_MPJPE={val_err*1000:.2f}mm")

    print(f"Stage2 best val MPJPE: {best_val*1000:.2f}mm -> {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Two-stage training: V4 pretrain + residual temporal fine-tune")
    parser.add_argument("--train", type=str, nargs="+", required=True, help="Train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Single-frame V4 pre-training epochs")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Temporal residual fine-tuning epochs")
    parser.add_argument("--stage1_train_samples", type=int, default=4000, help="Random single frames per train sequence")
    parser.add_argument("--stage2_train_samples", type=int, default=2000, help="Random clips per train sequence")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="If provided, skip stage 1 and load this V4 checkpoint")
    parser.add_argument("--stage1_output", type=str, default="outputs/ray_attention_v4_mpiinf3dhp_pretrain.pth")
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_residual_v2_mpiinf3dhp.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sample = np.load(args.train[0])
    n_views = sample["camera_K"].shape[0]
    j = sample["points_2d"].shape[2]
    print(f"n_views={n_views}, j={j}, clip_len={args.clip_len}, d={args.d}")

    # Stage 1 (or reuse existing checkpoint).
    if args.stage1_checkpoint:
        stage1_ckpt = args.stage1_checkpoint
        print(f"Skipping stage 1, using checkpoint: {stage1_ckpt}")
    else:
        stage1_ckpt = run_stage1(args, device, n_views, j)

    # Stage 2.
    run_stage2(args, device, n_views, j, stage1_ckpt)


if __name__ == "__main__":
    main()
