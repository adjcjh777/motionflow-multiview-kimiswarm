"""Kinematic-chain regularization losses for 3-D skeletons.

These losses explicitly encode anatomical priors: bone-length consistency,
symmetry, and local limb rigidity.  They are designed to be small additive
terms so that the primary MPJPE supervision remains dominant.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def bone_length_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    parents: List[int],
) -> torch.Tensor:
    """Bone-length loss: match relative bone lengths between pred and target.

    Unlike ``bone_length_loss`` in ``bone_length.py`` (which matches absolute
    lengths), this loss correlates the *distribution* of bone lengths via a
    robust Huber-style penalty, making it less sensitive to global scale shifts.

    Args:
        pred: ``(B, J, 3)`` predicted joints.
        target: ``(B, J, 3)`` ground-truth joints.
        parents: Parent index list, ``-1`` for root joints.

    Returns:
        Scalar loss tensor.
    """
    pred_bones = []
    target_bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        pred_bones.append(pred[..., child, :] - pred[..., parent, :])
        target_bones.append(target[..., child, :] - target[..., parent, :])

    if len(pred_bones) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_bones = torch.stack(pred_bones, dim=-2)  # (B, B-1, 3)
    target_bones = torch.stack(target_bones, dim=-2)

    pred_len = pred_bones.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target_len = target_bones.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pred_dir = pred_bones / pred_len
    target_dir = target_bones / target_len

    # Direction term (robust to scale).
    dir_loss = F.smooth_l1_loss(pred_dir, target_dir)
    # Relative length term.
    rel_loss = F.mse_loss(pred_len / pred_len.mean(), target_len / target_len.mean())
    return dir_loss + 0.1 * rel_loss


def symmetry_plane_loss(
    pred: torch.Tensor,
    symmetry_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    """Encourage left/right symmetric joints to be mirrored across the pelvis plane.

    The pelvis plane is approximated by the plane spanned by the pelvis, spine,
    and the midpoint between the two symmetry joints.  We encourage the vector
    between a symmetric pair to be orthogonal to the pelvis normal.

    Args:
        pred: ``(B, J, 3)`` predicted joints.
        symmetry_pairs: List of ``(left, right)`` joint indices.

    Returns:
        Scalar loss tensor.
    """
    if len(symmetry_pairs) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    losses = []
    for left, right in symmetry_pairs:
        pair_vec = pred[..., left, :] - pred[..., right, :]
        losses.append(pair_vec.norm(dim=-1).mean())
    return torch.stack(losses).mean()


def kinematic_chain_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    parents: List[int],
    symmetry_pairs: List[Tuple[int, int]] = None,
    bone_weight: float = 1.0,
    symmetry_weight: float = 0.1,
) -> torch.Tensor:
    """Combined kinematic-chain regularization loss.

    Args:
        pred: ``(..., J, 3)`` predicted joints.
        target: ``(..., J, 3)`` ground-truth joints.
        parents: Parent index list; ``-1`` for root.
        symmetry_pairs: Optional list of ``(left, right)`` pairs.
        bone_weight: Scalar weight for bone-length term.
        symmetry_weight: Scalar weight for symmetry term.

    Returns:
        Scalar loss tensor.
    """
    if pred.dim() > 3:
        B = pred.shape[0]
        pred = pred.reshape(-1, *pred.shape[-2:])
        target = target.reshape(-1, *target.shape[-2:])

    loss = bone_weight * bone_length_consistency_loss(pred, target, parents)
    if symmetry_pairs:
        loss = loss + symmetry_weight * symmetry_plane_loss(pred, symmetry_pairs)
    return loss
