"""Self-supervised multi-view clip datasets.

These datasets load canonical ``.npz`` files but do **not** require ``joints_3d``.
They are intended for masked-view reprojection pre-training of the temporal
ray-attention fusion models.

Expected ``.npz`` layout::

    points_2d   (T, V, J, 2)
    confidences (T, V, J)
    camera_K    (V, 3, 3)
    camera_R    (V, 3, 3)
    camera_t    (V, 3)
"""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class SSLTemporalClipDataset(Dataset):
    """Yield non-overlapping (or strided) clips without 3D ground truth."""

    def __init__(self, npz_path: str, clip_len: int, stride: int = 1):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()  # (T, V, J, 2)
        self.confidences = torch.from_numpy(data["confidences"]).float()  # (T, V, J)
        self.K = torch.from_numpy(data["camera_K"]).float()  # (V, 3, 3)
        self.R = torch.from_numpy(data["camera_R"]).float()  # (V, 3, 3)
        self.t = torch.from_numpy(data["camera_t"]).float()  # (V, 3)

        self.clip_len = clip_len
        self.stride = stride
        self.total_frames = self.points_2d.shape[0]
        self.num_clips = max(1, (self.total_frames - clip_len) // stride + 1)

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = idx * self.stride
        end = start + self.clip_len
        x = torch.cat(
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )  # (T, V, J, 3)
        return x, self.K, self.R, self.t


class SSLRandomClipDataset(Dataset):
    """Sample random clips without 3D ground truth for training augmentation."""

    def __init__(self, npz_path: str, clip_len: int, n_samples: int = 2000):
        data = np.load(npz_path)
        self.points_2d = torch.from_numpy(data["points_2d"]).float()
        self.confidences = torch.from_numpy(data["confidences"]).float()
        self.K = torch.from_numpy(data["camera_K"]).float()
        self.R = torch.from_numpy(data["camera_R"]).float()
        self.t = torch.from_numpy(data["camera_t"]).float()
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = self.points_2d.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start = random.randint(0, max(0, self.total_frames - self.clip_len))
        end = start + self.clip_len
        x = torch.cat(
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )
        return x, self.K, self.R, self.t


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Stack a list of (x, K, R, t) tuples into batches."""
    x = torch.stack([b[0] for b in batch], dim=0)
    K = torch.stack([b[1] for b in batch], dim=0)
    R = torch.stack([b[2] for b in batch], dim=0)
    t = torch.stack([b[3] for b in batch], dim=0)
    return x, K, R, t


def make_ssl_dataloaders(
    train_paths: List[str],
    val_path: str,
    clip_len: int,
    batch_size: int,
    train_samples: int = 4000,
    val_stride: int = 1,
):
    """Build train/val DataLoaders for SSL pre-training."""
    train_datasets = [
        SSLRandomClipDataset(p, clip_len, n_samples=train_samples) for p in train_paths
    ]
    val_dataset = SSLTemporalClipDataset(val_path, clip_len, stride=val_stride)

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return train_loader, val_loader
