"""Kinematic-chain auxiliary losses for v15."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


def joint_limit_loss(
    pred: torch.Tensor,
    parents: List[int],
    max_flexion_deg: float = 160.0,
) -> torch.Tensor:
    """Penalize interior joint angles exceeding ``max_flexion_deg``.

    For each interior joint we compute the angle between the incoming
    (parent -> joint) and outgoing (joint -> child) bones. Hinge joints
    (knees, elbows, ankles) cannot normally extend beyond ~160°, so we
    penalize ``angle > max_flexion_deg``.

    Args:
        pred: ``(..., J, 3)`` predicted joints. Accepts ``(B, T, J, 3)``
            or ``(B, J, 3)``.
        parents: Parent index list, ``-1`` for root joints.
        max_flexion_deg: Maximum allowed interior angle in degrees.

    Returns:
        Scalar loss tensor.
    """
    if pred.dim() == 4:  # (B, T, J, 3)
        B, T, J, _ = pred.shape
        pred = pred.reshape(B * T, J, 3)
    elif pred.dim() == 3:  # (B, J, 3)
        J = pred.shape[-2]
    else:
        raise ValueError(f"Unexpected pred shape {pred.shape}")

    cos_max = math.cos(math.radians(max_flexion_deg))
    losses = []

    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        children = [c for c, par in enumerate(parents) if par == j]
        for c in children:
            b_in = F.normalize(pred[..., j, :] - pred[..., p, :], dim=-1)
            b_out = F.normalize(pred[..., c, :] - pred[..., j, :], dim=-1)
            cos = (b_in * b_out).sum(dim=-1)
            losses.append(torch.relu(cos_max - cos))

    if not losses:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return torch.stack(losses).mean()


def temporal_bone_length_loss(
    pred: torch.Tensor,
    parents: List[int],
) -> torch.Tensor:
    """Penalize temporal variance of bone lengths (limb rigidity).

    Args:
        pred: ``(B, T, J, 3)`` predicted joints.
        parents: Parent index list, ``-1`` for root joints.

    Returns:
        Scalar loss tensor.
    """
    if pred.dim() != 4:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bones.append(pred[:, :, child, :] - pred[:, :, parent, :])
    if not bones:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    bones = torch.stack(bones, dim=-2)  # (B, T, n_bones, 3)
    lengths = bones.norm(dim=-1)  # (B, T, n_bones)
    return lengths.var(dim=1).mean()
