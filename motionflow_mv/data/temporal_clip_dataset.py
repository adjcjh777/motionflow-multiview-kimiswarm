"""Shared PyTorch datasets and helpers for training temporal fusion models.

This module contains the clip dataset classes and training utilities originally
implemented in ``experiments/train_ray_attention_temporal_mpiinf3dhp.py``.  They
are exposed here so that follow-up training scripts (e.g. for Shelf/Campus)
can reuse the same data loading / augmentation / evaluation logic without
copy-pasting code.

Expected canonical .npz layout::

    points_2d   (T, V, J, 2)
    confidences (T, V, J)
    joints_3d   (T, J, 3)
    camera_K    (V, 3, 3)
    camera_R    (V, 3, 3)
    camera_t    (V, 3)
"""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalClipDataset(Dataset):
    """Yield non-overlapping (or strided) clips from a long canonical .npz sequence."""

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
        y = self.joints_3d[start:end]  # (T, J, 3)
        return x, y, self.K, self.R, self.t


class RandomClipDataset(Dataset):
    """Sample random fixed-length clips from a sequence for training augmentation."""

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
        y = self.joints_3d[start:end]
        return x, y, self.K, self.R, self.t


def _apply_view_dropout(
    x: torch.Tensor,
    prob: float = 0.0,
    min_views: int = 2,
) -> torch.Tensor:
    """Zero out the confidence channel for randomly dropped views.

    Args:
        x: (B, T, V, J, 3) tensor of (x, y, confidence).
        prob: Probability of dropping each view independently.
        min_views: Minimum number of views to retain per batch element.

    Returns:
        x with confidence channel masked for dropped views.
    """
    if prob <= 0.0:
        return x
    B, T, V, J, _ = x.shape
    device = x.device
    keep = (torch.rand(B, V, device=device) > prob).float()
    for i in range(B):
        active = keep[i].nonzero(as_tuple=True)[0]
        if active.numel() < min_views:
            needed = min_views - active.numel()
            dropped = (keep[i] == 0).nonzero(as_tuple=True)[0]
            if dropped.numel() > 0:
                perm = torch.randperm(dropped.numel(), device=device)
                extra = dropped[perm[:needed]]
                keep[i, extra] = 1.0
    x[..., 2] = x[..., 2] * keep.view(B, 1, V, 1)
    return x


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Stack a list of (x, y, K, R, t) tuples into batches."""
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    return x, y, K, R, t


def augment_clip(
    x: torch.Tensor,
    noise_std: float = 0.5,
    dropout_rate: float = 0.1,
    outlier_rate: float = 0.02,
    outlier_scale: float = 100.0,
) -> torch.Tensor:
    """Lightweight per-clip augmentation.

    Operates in-place on a clone-safe manner (caller passes the tensor it wants
    to mutate).  Adds pixel noise, confidence dropout, and occasional 2D outliers.
    """
    if noise_std > 0:
        x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * noise_std
    if dropout_rate > 0:
        mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            > dropout_rate
        ).float()
        x[..., 2] = x[..., 2] * mask
    if outlier_rate > 0:
        outlier_mask = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
            < outlier_rate
        )
        outlier = (
            torch.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 2, device=x.device)
            - 0.5
        ) * 2 * outlier_scale
        x[..., :2] = torch.where(outlier_mask[..., None], outlier, x[..., :2])
    return x


def make_collate_fn_with_view_dropout(
    dropout_prob: float = 0.0,
    min_views: int = 2,
) -> callable:
    """Return a collate function that applies random whole-view dropout.

    Args:
        dropout_prob: Probability of dropping each view during training.
        min_views: Minimum number of views to retain per batch element.

    Returns:
        A collate function ``(batch) -> (x, y, K, R, t)`` that stacks the batch
        and then randomly zeros the confidence channel for dropped views.
    """
    def _collate(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        x, y, K, R, t = collate_fn(batch)
        x = _apply_view_dropout(x, prob=dropout_prob, min_views=min_views)
        return x, y, K, R, t

    return _collate


def make_dataloaders(
    train_paths: List[str],
    val_path: str,
    clip_len: int,
    batch_size: int,
    train_samples: int = 4000,
    view_dropout_prob: float = 0.0,
    view_dropout_min_views: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from canonical .npz paths."""
    train_datasets = [RandomClipDataset(p, clip_len, n_samples=train_samples) for p in train_paths]
    val_dataset = TemporalClipDataset(val_path, clip_len)

    train_collate = (
        make_collate_fn_with_view_dropout(view_dropout_prob, view_dropout_min_views)
        if view_dropout_prob > 0.0
        else collate_fn
    )

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return train_loader, val_loader
