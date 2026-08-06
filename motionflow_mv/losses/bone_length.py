"""Skeleton bone-length loss.

Enforces anatomical consistency by matching the lengths of each bone in the
predicted skeleton to those of the ground-truth skeleton.  The loss depends
only on a parent-index list, so it works for any joint ordering or skeleton
size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    parents: list[int],
) -> torch.Tensor:
    """L2 loss on bone lengths.

    Args:
        pred: Predicted 3D joints, shape (..., J, 3).
        target: Ground-truth 3D joints, shape (..., J, 3).
        parents: List of parent joint indices; -1 for the root(s).

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    j = pred.shape[-2]
    if len(parents) != j:
        raise ValueError(f"parents length ({len(parents)}) must equal J ({j})")

    # Compute bone vectors for all child-parent pairs.
    pred_bones = []
    target_bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        pred_bones.append(pred[..., child, :] - pred[..., parent, :])
        target_bones.append(target[..., child, :] - target[..., parent, :])

    if len(pred_bones) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_bones = torch.stack(pred_bones, dim=-2)  # (..., B, 3)
    target_bones = torch.stack(target_bones, dim=-2)

    pred_lengths = pred_bones.norm(dim=-1)
    target_lengths = target_bones.norm(dim=-1)

    return F.mse_loss(pred_lengths, target_lengths)
