"""Mixed-dataset loader for MPI-INF-3DHP, Human3.6M, and AIST++.

This module provides a single :class:`MixedDataset` that normalises clips from
multiple canonical ``.npz`` sources to a common view/joint grid so they can be
batched together.  Each sample is tagged with a dataset id so that models with
per-dataset heads (or dataset-specific normalisation) can dispatch correctly.

Expected canonical ``.npz`` layout::

    points_2d   (T, V, J, 2)
    confidences (T, V, J)
    joints_3d   (T, J, 3)
    camera_K    (V, 3, 3)
    camera_R    (V, 3, 3)
    camera_t    (V, 3)
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler


# Canonical dimensions inferred from MPI-INF-3DHP (the largest dataset).
DATASET_REGISTRY: Dict[str, Dict[str, int]] = {
    "mpi": {"id": 0, "n_views": 14, "n_joints": 28},
    "aist": {"id": 1, "n_views": 9, "n_joints": 17},
    "h36m": {"id": 2, "n_views": 4, "n_joints": 17},
}

DATASET_IDS: Dict[str, int] = {name: spec["id"] for name, spec in DATASET_REGISTRY.items()}
MAX_VIEWS = max(spec["n_views"] for spec in DATASET_REGISTRY.values())
MAX_JOINTS = max(spec["n_joints"] for spec in DATASET_REGISTRY.values())


def _pad_cameras(camera_K: np.ndarray, camera_R: np.ndarray, camera_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad camera arrays to ``MAX_VIEWS`` with identity/zero placeholders."""
    src_v = camera_K.shape[0]
    pad_v = MAX_VIEWS - src_v
    if pad_v <= 0:
        return camera_K, camera_R, camera_t

    K_pad = np.eye(3, dtype=camera_K.dtype)[None, ...].repeat(MAX_VIEWS, axis=0)
    R_pad = np.eye(3, dtype=camera_R.dtype)[None, ...].repeat(MAX_VIEWS, axis=0)
    t_pad = np.zeros((MAX_VIEWS, 3), dtype=camera_t.dtype)
    K_pad[:src_v] = camera_K
    R_pad[:src_v] = camera_R
    t_pad[:src_v] = camera_t
    return K_pad, R_pad, t_pad


class DatasetBalancedSampler(Sampler):
    """Sample equally from each sub-dataset inside a :class:`ConcatDataset`.

    Parameters
    ----------
    dataset_lengths:
        Number of samples in each sub-dataset.
    samples_per_dataset:
        How many samples to draw from each sub-dataset per epoch.  If ``None``,
        defaults to the length of the largest sub-dataset (with replacement
        for smaller datasets).
    replacement:
        If ``True`` (default), smaller datasets are oversampled so that every
        epoch contains the same number of samples from each dataset.  If
        ``False``, the epoch size is capped at the smallest sub-dataset and
        all samples are drawn without replacement.
    seed:
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        dataset_lengths: Sequence[int],
        samples_per_dataset: Optional[int] = None,
        replacement: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset_lengths = list(dataset_lengths)
        if any(n <= 0 for n in self.dataset_lengths):
            raise ValueError("All dataset lengths must be positive.")

        max_len = max(self.dataset_lengths)
        self.samples_per_dataset = samples_per_dataset if samples_per_dataset is not None else max_len
        self.replacement = replacement

        if not replacement:
            self.samples_per_dataset = min(self.samples_per_dataset, min(self.dataset_lengths))

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_dataset * len(self.dataset_lengths)

    def __iter__(self):
        indices = []
        offset = 0
        for length in self.dataset_lengths:
            if self.replacement:
                sample_idx = self._rng.integers(0, length, size=self.samples_per_dataset)
            else:
                sample_idx = self._rng.choice(length, size=self.samples_per_dataset, replace=False)
            indices.extend(offset + sample_idx)
            offset += length
        return iter(indices)


class MixedDataset(Dataset):
    """Yield padded clips from a canonical ``.npz`` and tag them by dataset id.

    Parameters
    ----------
    npz_path:
        Path to a canonical multi-view ``.npz`` file.
    dataset_name:
        One of ``"mpi"``, ``"aist"``, ``"h36m"``.
    clip_len:
        Number of frames in each temporal clip.
    n_samples:
        If ``None``, yield deterministic strided clips.  If an integer,
        sample that many random clips with replacement from the sequence.
    stride:
        Stride between consecutive clips when ``n_samples`` is ``None``.
    """

    def __init__(
        self,
        npz_path: str,
        dataset_name: str,
        clip_len: int,
        n_samples: Optional[int] = None,
        stride: int = 1,
    ):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(DATASET_REGISTRY.keys())}"
            )

        data = np.load(npz_path)
        src_v = int(data["camera_K"].shape[0])
        src_j = int(data["joints_3d"].shape[1])

        self.dataset_name = dataset_name
        self.dataset_id = DATASET_IDS[dataset_name]
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = int(data["points_2d"].shape[0])

        if src_v > MAX_VIEWS:
            raise ValueError(
                f"{dataset_name} has {src_v} views, exceeding MAX_VIEWS={MAX_VIEWS}"
            )
        if src_j > MAX_JOINTS:
            raise ValueError(
                f"{dataset_name} has {src_j} joints, exceeding MAX_JOINTS={MAX_JOINTS}"
            )

        # Pad to common (MAX_VIEWS, MAX_JOINTS) grid.
        pad_v = MAX_VIEWS - src_v
        pad_j = MAX_JOINTS - src_j

        points_2d = data["points_2d"]
        confidences = data["confidences"]
        joints_3d = data["joints_3d"]

        if pad_v or pad_j:
            points_2d = np.pad(
                points_2d,
                ((0, 0), (0, max(0, pad_v)), (0, max(0, pad_j)), (0, 0)),
            )
            confidences = np.pad(
                confidences,
                ((0, 0), (0, max(0, pad_v)), (0, max(0, pad_j))),
            )
        if pad_j:
            joints_3d = np.pad(joints_3d, ((0, 0), (0, pad_j), (0, 0)))

        camera_K, camera_R, camera_t = _pad_cameras(
            data["camera_K"], data["camera_R"], data["camera_t"]
        )

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

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
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


def mixed_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Collate a list of mixed-dataset samples into a batch."""
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, dataset_ids


def build_mixed_dataloaders(
    train_paths: Dict[str, Sequence[str]],
    val_path: str,
    val_dataset: str,
    clip_len: int,
    batch_size: int,
    train_samples: int = 500,
    val_stride: int = 1,
    num_workers: int = 0,
    balance_datasets: bool = False,
    balance_samples_per_dataset: Optional[int] = None,
    balance_seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build mixed-dataset train/val :class:`DataLoader` instances.

    Parameters
    ----------
    train_paths:
        Mapping from dataset name to a list of ``.npz`` paths used for training.
    val_path:
        Path to the validation ``.npz`` file.
    val_dataset:
        Dataset name for the validation file.
    clip_len:
        Number of frames per clip.
    batch_size:
        Batch size for both loaders.
    train_samples:
        Number of random clips to sample from each training sequence.
    val_stride:
        Stride for validation clips.
    balance_datasets:
        If ``True``, use :class:`DatasetBalancedSampler` so that each epoch
        samples equally from every training dataset instead of proportionally
        to dataset size.
    balance_samples_per_dataset:
        Optional override for the number of samples drawn per dataset when
        ``balance_datasets`` is ``True``.
    balance_seed:
        Optional seed for the balanced sampler.

    Returns
    -------
    train_loader, val_loader
    """
    train_datasets: List[Dataset] = []
    for dataset_name, paths in train_paths.items():
        for p in paths:
            train_datasets.append(
                MixedDataset(p, dataset_name, clip_len, n_samples=train_samples)
            )
    val_dataset_obj = MixedDataset(
        val_path, val_dataset, clip_len, n_samples=None, stride=val_stride
    )

    if balance_datasets:
        sampler = DatasetBalancedSampler(
            [len(ds) for ds in train_datasets],
            samples_per_dataset=balance_samples_per_dataset,
            seed=balance_seed,
        )
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=mixed_collate_fn,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mixed_collate_fn,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
        )
    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mixed_collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    return train_loader, val_loader
