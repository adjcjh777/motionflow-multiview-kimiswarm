"""Canonical-skeleton consistency loss.

Encourages the predicted 3D pose to respect the same skeleton topology as the
ground truth, regardless of which dataset the sample came from.  This makes the
model's output live in a unified canonical skeleton space across datasets.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def _bone_vectors(joints: torch.Tensor, parents: List[int]) -> torch.Tensor:
    """Return bone vectors for every child-parent edge.

    Parameters
    ----------
    joints:
        ``(..., J, 3)`` 3D joints.
    parents:
        Parent index for each joint; ``-1`` for roots.

    Returns
    -------
    ``(..., B, 3)`` bone vectors, where B is the number of bones with a parent.
    """
    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bones.append(joints[..., child, :] - joints[..., parent, :])
    return torch.stack(bones, dim=-2)


def canonical_skeleton_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    parents: List[int],
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bone-length consistency loss in canonical skeleton space.

    Parameters
    ----------
    pred:
        ``(..., J, 3)`` predicted 3D joints.
    target:
        ``(..., J, 3)`` ground-truth 3D joints.
    parents:
        Parent indices defining the canonical skeleton topology.
    mask:
        Optional ``(..., J)`` boolean mask for valid joints.  If provided,
        invalid joints are excluded from the length computation.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    Scalar MSE between predicted and target bone lengths.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

    pred_bones = _bone_vectors(pred, parents)
    target_bones = _bone_vectors(target, parents)

    pred_lengths = pred_bones.norm(dim=-1)
    target_lengths = target_bones.norm(dim=-1)

    if mask is not None:
        # A bone is valid only if both its child and parent are valid.
        bone_mask = []
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            bone_mask.append(mask[..., child] & mask[..., parent])
        bone_mask = torch.stack(bone_mask, dim=-1).float()
        loss = (F.mse_loss(pred_lengths, target_lengths, reduction="none") * bone_mask).sum()
        loss = loss / (bone_mask.sum() + eps)
        return loss

    return F.mse_loss(pred_lengths, target_lengths)


def canonical_bone_length_regularizer(
    pred: torch.Tensor,
    canonical_lengths: torch.Tensor,
    parents: List[int],
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Regularize predicted bone lengths toward a fixed canonical prior.

    Parameters
    ----------
    pred:
        ``(..., J, 3)`` predicted 3D joints.
    canonical_lengths:
        ``(B,)`` or ``(...)`` tensor of expected bone lengths.  Must broadcast
        with the computed predicted bone lengths.
    parents:
        Parent indices defining the canonical skeleton topology.
    mask:
        Optional boolean mask of valid joints.
    eps:
        Small constant.

    Returns
    -------
    Scalar MSE between predicted and canonical bone lengths.
    """
    pred_bones = _bone_vectors(pred, parents)
    pred_lengths = pred_bones.norm(dim=-1)
    if mask is not None:
        bone_mask = []
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            bone_mask.append(mask[..., child] & mask[..., parent])
        bone_mask = torch.stack(bone_mask, dim=-1).float()
        loss = (F.mse_loss(pred_lengths, canonical_lengths, reduction="none") * bone_mask).sum()
        loss = loss / (bone_mask.sum() + eps)
        return loss
    return F.mse_loss(pred_lengths, canonical_lengths)


if __name__ == "__main__":
    torch.manual_seed(0)
    J = 17
    parents = H36M_17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 16]
    pred = torch.randn(2, 10, J, 3)
    target = torch.randn(2, 10, J, 3)
    loss = canonical_skeleton_loss(pred, target, parents)
    assert torch.isfinite(loss)
    print(f"canonical_skeleton_loss = {loss.item():.4f}")

    # Masked variant.
    mask = torch.rand(2, 10, J) > 0.1
    masked_loss = canonical_skeleton_loss(pred, target, parents, mask=mask)
    assert torch.isfinite(masked_loss)
    print(f"masked canonical_skeleton_loss = {masked_loss.item():.4f}")
    print("canonical_skeleton_loss smoke test passed")
