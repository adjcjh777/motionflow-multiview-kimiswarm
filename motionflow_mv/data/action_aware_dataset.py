"""Action-aware multi-view clip datasets for Human3.6M.

The datasets below extend the canonical ``.npz`` loaders by returning an action
label/embedding for each clip.  Action ids are parsed from the filename so that
the caller does not have to remember the mapping.

Expected canonical .npz layout::

    points_2d   (T, V, J, 2)
    confidences (T, V, J)
    joints_3d   (T, J, 3)
    camera_K    (V, 3, 3)
    camera_R    (V, 3, 3)
    camera_t    (V, 3)

Author: research swarm (direction 17 – action semantics)
"""

import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


# Human3.6M action ids used by the WebBridge preprocessing (2-16).
# Names follow the official Human3.6M convention.
ACTION_NAMES: Dict[int, str] = {
    1: "Directions",
    2: "Discussion",
    3: "Eating",
    4: "Greeting",
    5: "Phoning",
    6: "Posing",
    7: "Purchases",
    8: "Sitting",
    9: "SittingDown",
    10: "Smoking",
    11: "Photo",
    12: "Waiting",
    13: "Walking",
    14: "WalkDog",
    15: "WalkTogether",
    16: "Discussion",  # legacy duplicate guard, should not happen in normal flow
    17: "Directions",  # guard
}


def extract_action_id(npz_path: str, strict: bool = False) -> int:
    """Parse the action id from a Human3.6M ``.npz`` filename.

    Supported filename patterns::

        s_01_act_02_multiview.npz
        s_01_acts_02_multiview.npz
        s_01_acts_02_03_04_multiview.npz

    Args:
        npz_path: path to the npz file.
        strict: if True, raise an error when multiple actions are present;
            otherwise use the first action.

    Returns:
        The parsed action id.

    Raises:
        ValueError: if no action id can be parsed.
    """
    stem = npz_path.split("/")[-1].split("\\")[-1].replace(".npz", "")
    match = re.search(r"(?:act|acts)_([0-9]+(?:_[0-9]+)*)_?", stem)
    if not match:
        raise ValueError(f"Could not parse action id from {npz_path}")
    actions = [int(a) for a in match.group(1).split("_")]
    if len(actions) > 1 and strict:
        raise ValueError(
            f"Multi-action file {npz_path} contains {actions}; "
            "pass `action_id` explicitly or set strict=False"
        )
    return actions[0]


class ActionAwareTemporalClipDataset(Dataset):
    """Yield non-overlapping (or strided) clips with an action label."""

    def __init__(
        self,
        npz_path: str,
        clip_len: int,
        stride: int = 1,
        action_id: Optional[int] = None,
    ):
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

        self.action_id = action_id if action_id is not None else extract_action_id(npz_path)

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
        return x, y, self.K, self.R, self.t, self.action_id


class ActionAwareRandomClipDataset(Dataset):
    """Sample random fixed-length clips with an action label."""

    def __init__(
        self,
        npz_path: str,
        clip_len: int,
        n_samples: int = 2000,
        action_id: Optional[int] = None,
    ):
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

        self.action_id = action_id if action_id is not None else extract_action_id(npz_path)

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
        return x, y, self.K, self.R, self.t, self.action_id


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """Stack a list of (x, y, K, R, t, action) tuples into batches."""
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    K = torch.stack([b[2] for b in batch], dim=0)
    R = torch.stack([b[3] for b in batch], dim=0)
    t = torch.stack([b[4] for b in batch], dim=0)
    action = torch.tensor([b[5] for b in batch], dtype=torch.long)
    return x, y, K, R, t, action


def make_action_aware_dataloaders(
    train_paths: List[str],
    val_path: str,
    clip_len: int,
    batch_size: int,
    train_samples: int = 4000,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train/val DataLoaders from canonical .npz paths."""
    train_datasets = [
        ActionAwareRandomClipDataset(p, clip_len, n_samples=train_samples) for p in train_paths
    ]
    val_dataset = ActionAwareTemporalClipDataset(val_path, clip_len)

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


def action_distribution(paths: List[str], normalize: bool = False) -> Dict[int, int]:
    """Count frames per action id across a list of canonical .npz files.

    Args:
        paths: list of .npz paths.
        normalize: if True, return probabilities instead of counts.

    Returns:
        Mapping from action id to frame count (or probability).
    """
    counts: Dict[int, int] = {}
    for path in paths:
        action_id = extract_action_id(path)
        data = np.load(path)
        n_frames = data["points_2d"].shape[0]
        counts[action_id] = counts.get(action_id, 0) + n_frames
    if normalize:
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total > 0 else counts
    return counts
