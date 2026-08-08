"""WebBridge mixed-dataset loader with a unified 17-joint canonical skeleton.

The loader consumes canonical ``.npz`` files from any WebBridge source
(H36M, MPI-INF-3DHP, AIST++, Shelf, Campus), re-indexes the joint
skeleton to a common 17-joint layout, pads views to the largest rig
(14 views for MPI-INF-3DHP), and yields temporal clips.

Expected canonical ``.npz`` layout::

    points_2d   (T, V, J, 2)
    confidences (T, V, J)
    joints_3d   (T, J, 3)
    camera_K    (V, 3, 3)
    camera_R    (V, 3, 3)
    camera_t    (V, 3)

Each sample returns ``(x, y, K, R, t, dataset_id)`` where ``x`` has shape
``(T, MAX_VIEWS, 17, 3)`` and ``y`` has shape ``(T, 17, 3)``.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


# Largest WebBridge rig is MPI-INF-3DHP with 14 calibrated views.
MAX_VIEWS = 14

# Common 17-joint skeleton (H36M ordering).
CANONICAL_17_JOINTS = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "head_top",
]

# Mapping from each WebBridge source to the canonical 17-joint layout.
# Values are length-17 index arrays: canonical_joint = source_joint[map].
SKELETON_MAPS: Dict[str, np.ndarray] = {
    # H36M and AIST++ already use the same 17-joint layout.
    "h36m": np.arange(17, dtype=np.int64),
    "aist": np.arange(17, dtype=np.int64),
    # Shelf/Campus are converted to the H36M 17-joint layout.
    "shelf": np.arange(17, dtype=np.int64),
    "campus": np.arange(17, dtype=np.int64),
    # MPI-INF-3DHP 28-joint -> canonical 17-joint mapping.
    # Source joint name order (from MPI_INF_3DHP_28_PARENTS):
    #   0:spine3, 1:spine4, 2:spine2, 3:spine, 4:pelvis, 5:neck,
    #   6:head, 7:head_top, 8:left_clavicle, 9:left_shoulder,
    #   10:left_elbow, 11:left_wrist, 12:left_hand, 13:right_clavicle,
    #   14:right_shoulder, 15:right_elbow, 16:right_wrist, 17:right_hand,
    #   18:left_hip, 19:left_knee, 20:left_ankle, 21:left_foot,
    #   22:left_toe, 23:right_hip, 24:right_knee, 25:right_ankle,
    #   26:right_foot, 27:right_toe
    "mpi": np.array(
        [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 9, 10, 11, 14, 15, 16, 7],
        dtype=np.int64,
    ),
}

# Stable dataset IDs used by the loader.
DATASET_IDS: Dict[str, int] = {
    "h36m": 0,
    "mpi": 1,
    "aist": 2,
    "shelf": 3,
    "campus": 4,
}


def _pad_cameras(
    camera_K: np.ndarray,
    camera_R: np.ndarray,
    camera_t: np.ndarray,
    max_views: int = MAX_VIEWS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad camera arrays to ``max_views`` with identity/zero placeholders."""
    src_v = camera_K.shape[0]
    if src_v > max_views:
        raise ValueError(f"{src_v} views exceeds MAX_VIEWS={max_views}")
    if src_v == max_views:
        return camera_K.copy(), camera_R.copy(), camera_t.copy()

    K_pad = np.eye(3, dtype=camera_K.dtype)[None, ...].repeat(max_views, axis=0)
    R_pad = np.eye(3, dtype=camera_R.dtype)[None, ...].repeat(max_views, axis=0)
    t_pad = np.zeros((max_views, 3), dtype=camera_t.dtype)
    K_pad[:src_v] = camera_K
    R_pad[:src_v] = camera_R
    t_pad[:src_v] = camera_t
    return K_pad, R_pad, t_pad


def _reindex_and_pad_views(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    joint_map: np.ndarray,
    max_views: int = MAX_VIEWS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-index joints and pad views to ``max_views``."""
    points_2d = points_2d[..., joint_map, :]
    confidences = confidences[..., joint_map]
    joints_3d = joints_3d[:, joint_map, :]

    src_v = points_2d.shape[1]
    pad_v = max_views - src_v
    if pad_v > 0:
        points_2d = np.pad(points_2d, ((0, 0), (0, pad_v), (0, 0), (0, 0)))
        confidences = np.pad(confidences, ((0, 0), (0, pad_v), (0, 0)))
    return points_2d, confidences, joints_3d


class WebBridgeCanonical17Dataset(Dataset):
    """Load one canonical ``.npz``, map to the canonical 17-joint skeleton, and
    yield padded temporal clips.

    Parameters
    ----------
    npz_path:
        Path to a canonical WebBridge ``.npz`` file.
    dataset_name:
        One of ``"h36m"``, ``"mpi"``, ``"aist"``, ``"shelf"``, ``"campus"``.
    clip_len:
        Number of frames per temporal clip.
    n_samples:
        If ``None``, yield deterministic strided clips.  If an integer,
        sample that many random clips with replacement.
    stride:
        Stride between consecutive clips when ``n_samples`` is ``None``.
    return_view_mask:
        If ``True``, each sample also returns a ``(MAX_VIEWS,)`` boolean mask
        with ``True`` entries for the real (non-padded) views.  This is
        required by geometry-aware models such as
        :class:`MultiViewGeometryFusionV25` so that padded views are ignored
        during cross-view attention and depth-proposal triangulation.
    """

    def __init__(
        self,
        npz_path: str,
        dataset_name: str,
        clip_len: int,
        n_samples: Optional[int] = None,
        stride: int = 1,
        return_view_mask: bool = False,
    ):
        if dataset_name not in SKELETON_MAPS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(SKELETON_MAPS.keys())}"
            )

        data = np.load(npz_path)
        joint_map = SKELETON_MAPS[dataset_name]
        src_j = int(data["joints_3d"].shape[1])
        src_v = int(data["points_2d"].shape[1])
        if joint_map.max() >= src_j:
            raise ValueError(
                f"Skeleton map for {dataset_name} references joint {joint_map.max()} "
                f"but source only has {src_j} joints."
            )

        self.dataset_name = dataset_name
        self.dataset_id = DATASET_IDS[dataset_name]
        self.clip_len = clip_len
        self.n_samples = n_samples
        self.total_frames = int(data["points_2d"].shape[0])
        self.n_views = src_v
        self.return_view_mask = return_view_mask

        points_2d, confidences, joints_3d = _reindex_and_pad_views(
            data["points_2d"],
            data["confidences"],
            data["joints_3d"],
            joint_map,
        )
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
            [
                self.points_2d[start:end],
                self.confidences[start:end].unsqueeze(-1),
            ],
            dim=-1,
        )  # (T, V, J, 3)
        y = self.joints_3d[start:end]  # (T, J, 3)
        if self.return_view_mask:
            view_mask = torch.zeros(MAX_VIEWS, dtype=torch.bool)
            view_mask[: self.n_views] = True
            return x, y, self.camera_K, self.camera_R, self.camera_t, self.dataset_id, view_mask
        return x, y, self.camera_K, self.camera_R, self.camera_t, self.dataset_id


class WebBridgeMixedDataset(ConcatDataset):
    """Concatenate multiple :class:`WebBridgeCanonical17Dataset` instances."""

    def __init__(
        self,
        npz_paths: Sequence[str],
        dataset_names: Sequence[str],
        clip_len: int,
        n_samples: Optional[int] = None,
        stride: int = 1,
        return_view_mask: bool = False,
    ):
        if len(npz_paths) != len(dataset_names):
            raise ValueError("npz_paths and dataset_names must have the same length")
        datasets = [
            WebBridgeCanonical17Dataset(p, name, clip_len, n_samples, stride, return_view_mask)
            for p, name in zip(npz_paths, dataset_names)
        ]
        super().__init__(datasets)


def webbridge_mixed_collate_fn(
    batch: List[Tuple],
) -> Tuple[torch.Tensor, ...]:
    """Collate a list of mixed-dataset samples into a batch."""
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, dataset_ids


def webbridge_mixed_collate_fn_with_mask(
    batch: List[Tuple],
) -> Tuple[torch.Tensor, ...]:
    """Collate mixed-dataset samples that include a per-sample view mask.

    The view mask is stacked to ``(B, MAX_VIEWS)`` and returned as the last
    element.  It should be expanded to ``(B, T, V)`` by the consumer.
    """
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    dataset_ids = torch.tensor([b[5] for b in batch], dtype=torch.long)
    view_mask = torch.stack([b[6] for b in batch], dim=0)
    return x, y, K, R, t, dataset_ids, view_mask


def build_webbridge_mixed_dataloaders(
    train_paths: Sequence[str],
    train_names: Sequence[str],
    val_paths: Sequence[str],
    val_names: Sequence[str],
    clip_len: int,
    batch_size: int,
    train_samples: int = 200,
    val_stride: int = 1,
    num_workers: int = 0,
    return_view_mask: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for mixed WebBridge data.

    Parameters
    ----------
    train_paths, train_names:
        Parallel sequences of training ``.npz`` paths and dataset names.
    val_paths, val_names:
        Parallel sequences of validation ``.npz`` paths and dataset names.
    clip_len:
        Frames per clip.
    batch_size:
        Batch size for both loaders.
    train_samples:
        Random clips sampled per training sequence.
    val_stride:
        Stride for validation clips.
    num_workers:
        ``DataLoader`` workers.
    return_view_mask:
        If ``True``, loaders return an additional per-sample view mask.

    Returns
    -------
    train_loader, val_loader
    """
    train_dataset = WebBridgeMixedDataset(
        train_paths, train_names, clip_len, n_samples=train_samples,
        return_view_mask=return_view_mask,
    )
    val_dataset = WebBridgeMixedDataset(
        val_paths, val_names, clip_len, n_samples=None, stride=val_stride,
        return_view_mask=return_view_mask,
    )
    collate_fn = (
        webbridge_mixed_collate_fn_with_mask
        if return_view_mask
        else webbridge_mixed_collate_fn
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
    return train_loader, val_loader
