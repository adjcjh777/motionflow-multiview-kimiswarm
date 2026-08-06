"""Train RayAttentionFusionModelTemporalMixedResidual on MPI + AIST++ + H36M.

Mixed-dataset smoke training with a shared temporal backbone and per-dataset
output/residual heads.  Each canonical .npz file is padded to the largest
(MPI-INF-3DHP) view/joint grid, so clips from different datasets can be batched
together.

Example
-------
    conda run -n mf python experiments/train_ray_attention_temporal_mixed_residual_v1.py \
        --mpi_train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
        --aist_train data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz \
        --h36m_train data/h36m_hf/s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --val_dataset mpi \
        --clip_len 13 --epochs 2 --d 32 --train_samples 500
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

from motionflow_mv.fusion.ray_attention_temporal_mixed_residual_v1 import (
    RayAttentionFusionModelTemporalMixedResidual,
)


# Shared canonical dimensions inferred from MPI-INF-3DHP (the largest dataset).
MAX_VIEWS = 14
MAX_JOINTS = 28

DATASET_IDS = {"mpi": 0, "aist": 1, "h36m": 2}


class MixedTemporalDataset(torch.utils.data.Dataset):
    """Yield padded clips from a canonical .npz and tag them by dataset id."""

    def __init__(
        self,
        npz_path: str,
        dataset_name: str,
        clip_len: int,
        n_samples: int | None = None,
        stride: int = 1,
    ):
        data = np.load(npz_path)
        src_v = data["camera_K"].shape[0]
        src_j = data["joints_3d"].shape[1]

        self.dataset_id = DATASET_IDS[dataset_name]
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = data["points_2d"].shape[0]

        # Pad to common (MAX_VIEWS, MAX_JOINTS) grid.
        pad_v = MAX_VIEWS - src_v
        pad_j = MAX_JOINTS - src_j

        points_2d = data["points_2d"]
        confidences = data["confidences"]
        joints_3d = data["joints_3d"]

        if pad_v or pad_j:
            points_2d = np.pad(points_2d, ((0, 0), (0, pad_v), (0, pad_j), (0, 0)))
            confidences = np.pad(confidences, ((0, 0), (0, pad_v), (0, pad_j)))
        if pad_j:
            joints_3d = np.pad(joints_3d, ((0, 0), (0, pad_j), (0, 0)))

        # Camera padding: identity/zero for dummy views.
        camera_K = np.eye(3, dtype=np.float64)[None, ...].repeat(MAX_VIEWS, axis=0)
        camera_R = np.eye(3, dtype=np.float64)[None, ...].repeat(MAX_VIEWS, axis=0)
        camera_t = np.zeros((MAX_VIEWS, 3), dtype=np.float64)
        camera_K[:src_v] = data["camera_K"]
        camera_R[:src_v] = data["camera_R"]
        camera_t[:src_v] = data["camera_t"]

        self.points_2d = torch.from_numpy(points_2d).float()
        self.confidences = torch.from_numpy(confidences).float()
        self.joints_3d = torch.from_numpy(joints_3d).float()
        self.camera_K = torch.from_numpy(camera_K).float()
        self.camera_R = torch.from_numpy(camera_R).float()
        self.camera_t = torch.from_numpy(camera_t).float()

        if n_samples is None:
            self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)
            self.stride = stride
        else:
            self.num_clips = n_samples
            self.stride = 1

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        if self.n_samples is not None:
            start = random.randint(0, max(0, self.total_frames - self.clip_len))
        else:
            start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [self.points_2d[start:end], self.confidences[start:end].unsqueeze(-1)],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.camera_K, self.camera_R, self.camera_t, self.dataset_id


def collate_fn(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, dataset_ids


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_clip(x, noise_std: float = 0.5, dropout_rate: float = 0.1):
    """Lightweight per-clip augmentation."""
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            > dropout_rate
        ).float()
        x[..., 2] = x[..., 2] * mask
    return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err = 0.0
    total_count = 0
    for xb, yb, K, R, t, ids in loader:
        xb, yb = xb.to(device), yb.to(device)
        K, R, t = K.to(device), R.to(device), t.to(device)
        ids = ids.to(device)
        pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=ids)
        err = (pred - yb).norm(dim=-1) * mask.float()  # (B, T, J)
        total_err += err.sum().item()
        total_count += mask.sum().item()
    return total_err / total_count


def main():
    parser = argparse.ArgumentParser(
        description="Train mixed-dataset temporal ray-attention fusion with residual refinement"
    )
    parser.add_argument("--mpi_train", type=str, nargs="+", default=[], help="MPI-INF-3DHP train .npz files")
    parser.add_argument("--aist_train", type=str, nargs="+", default=[], help="AIST++ train .npz files")
    parser.add_argument("--h36m_train", type=str, nargs="+", default=[], help="Human3.6M train .npz files")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--val_dataset", type=str, required=True, choices=list(DATASET_IDS.keys()))
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=500, help="Random clips per train sequence")
    parser.add_argument("--val_samples", type=int, default=None, help="If set, sample this many random clips for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ray_attention_temporal_mixed_residual_v1.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build train/val datasets.
    train_datasets = []
    for p in args.mpi_train:
        train_datasets.append(MixedTemporalDataset(p, "mpi", args.clip_len, n_samples=args.train_samples))
    for p in args.aist_train:
        train_datasets.append(MixedTemporalDataset(p, "aist", args.clip_len, n_samples=args.train_samples))
    for p in args.h36m_train:
        train_datasets.append(MixedTemporalDataset(p, "h36m", args.clip_len, n_samples=args.train_samples))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = MixedTemporalDataset(
        args.val, args.val_dataset, args.clip_len,
        n_samples=args.val_samples, stride=1 if args.val_samples is None else 1,
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

    model = RayAttentionFusionModelTemporalMixedResidual(
        d=args.d,
        n_temporal_layers=args.n_temporal_layers,
        residual_hidden=args.residual_hidden,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb, K, R, t, ids in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            ids = ids.to(device)
            xb = augment_clip(xb)

            pred, mask = model(xb, K=K, R=R, t=t, dataset_ids=ids)
            loss = (((pred - yb) ** 2).sum(dim=-1) * mask.float()).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mask.sum().item()
            train_count += mask.sum().item()

        train_loss /= train_count
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
