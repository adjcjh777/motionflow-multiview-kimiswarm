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
from typing import List, Tuple, Union

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


def mask_views(
    x: torch.Tensor,
    ratio: float = 0.25,
    mode: str = "mixed",
    ensure_min_views: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask random views/time steps by zeroing confidences.

    Args:
        x: (B, T, V, J, 3) with the last channel as confidence.
        ratio: fraction of slots to mask.
        mode: ``view``, ``time``, or ``mixed``.
        ensure_min_views: minimum number of unmasked views per frame.

    Returns:
        x_masked, mask where ``mask`` is a ``(B, T, V, J)`` boolean tensor with
        ``True`` for masked slots. Confidences of masked slots are set to zero.
    """
    if x.dim() != 5:
        raise ValueError(f"mask_views expects (B,T,V,J,3), got {x.shape}")
    B, T, V, J, _ = x.shape
    masked = torch.zeros(B, T, V, J, dtype=torch.bool, device=x.device)

    if mode in ("view", "mixed"):
        n_view_mask = (
            max(1, int(V * ratio / 2)) if mode == "mixed" else max(1, int(V * ratio))
        )
        k = min(n_view_mask, max(0, V - ensure_min_views))
        if k > 0:
            for b in range(B):
                for t_idx in range(T):
                    idx = torch.randperm(V, device=x.device)[:k]
                    masked[b, t_idx, idx, :] = True

    if mode in ("time", "mixed"):
        n_time_mask = (
            max(1, int(T * ratio / 2)) if mode == "mixed" else max(1, int(T * ratio))
        )
        k = min(n_time_mask, max(0, T - 1))
        if k > 0:
            for b in range(B):
                idx = torch.randperm(T, device=x.device)[:k]
                masked[b, idx, :, :] = True

    x_masked = x.clone()
    x_masked[..., 2] = x_masked[..., 2] * (~masked).float()
    return x_masked, masked


class MaskedViewReprojectionDataset(Dataset):
    """Wrap an SSL clip dataset and apply masked-view reprojection on the fly.

    Each sample returns ``(x_masked, mask, x_original, K, R, t)`` so that a
    training loop can compute reprojection loss on both visible and masked
    slots without leaking ground-trread confidences into the masked slots.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        mask_ratio: float = 0.25,
        mask_mode: str = "mixed",
    ):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x, K, R, t = self.base_dataset[idx]
        x_masked, mask = mask_views(
            x.unsqueeze(0), self.mask_ratio, self.mask_mode, ensure_min_views=1
        )
        x_masked = x_masked.squeeze(0)
        mask = mask.squeeze(0)
        return x_masked, mask, x, K, R, t


def collate_masked_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Stack ``(x_masked, mask, x_original, K, R, t)`` tuples into batches."""
    x_masked = torch.stack([b[0] for b in batch], dim=0)
    mask = torch.stack([b[1] for b in batch], dim=0)
    x = torch.stack([b[2] for b in batch], dim=0)
    K = torch.stack([b[3] for b in batch], dim=0)
    R = torch.stack([b[4] for b in batch], dim=0)
    t = torch.stack([b[5] for b in batch], dim=0)
    return x_masked, mask, x, K, R, t


def make_ssl_dataloaders_with_masking(
    train_paths: List[str],
    val_path: str,
    clip_len: int,
    batch_size: int,
    mask_ratio: float = 0.25,
    mask_mode: str = "mixed",
    train_samples: int = 4000,
    val_stride: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val DataLoaders that yield masked-view reprojection samples.

    Returns:
        train_loader, val_loader. Each batch is ``(x_masked, mask, x, K, R, t)``.
    """
    train_base = [
        SSLRandomClipDataset(p, clip_len, n_samples=train_samples) for p in train_paths
    ]
    train_datasets = [MaskedViewReprojectionDataset(d, mask_ratio, mask_mode) for d in train_base]

    val_base = SSLTemporalClipDataset(val_path, clip_len, stride=val_stride)
    val_dataset = MaskedViewReprojectionDataset(val_base, mask_ratio, mask_mode)

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_masked_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_masked_fn,
        num_workers=0,
    )
    return train_loader, val_loader
