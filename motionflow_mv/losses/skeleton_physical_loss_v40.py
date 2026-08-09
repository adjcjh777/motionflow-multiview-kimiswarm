"""v40: Skeleton-aware physical loss.

Combines bone-length matching, soft joint-angle limits, left/right symmetry,
and a foot-floor contact prior.  All terms are optional and can be linearly
warm-up-ed to avoid dominating early training.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bone_vectors(pred: torch.Tensor, parents: List[int]) -> Tuple[torch.Tensor, List[int]]:
    """Return stacked bone vectors and the list of child indices."""
    children = []
    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        children.append(child)
        bones.append(pred[..., child, :] - pred[..., parent, :])
    return torch.stack(bones, dim=-2), children  # (..., n_bones, 3)


class SkeletonPhysicalLossV40(nn.Module):
    """Composite skeleton physical loss.

    Parameters
    ----------
    parents:
        Parent index list, ``-1`` for root joints.
    symmetry_pairs:
        List of ``(left, right)`` joint index pairs.
    foot_indices:
        List of foot joint indices for the floor-contact term.
    bone_weight:
        Weight for bone-length matching against ground truth.
    joint_limit_weight:
        Weight for soft joint-angle limit penalty.
    symmetry_weight:
        Weight for left/right symmetry penalty on bone lengths.
    floor_weight:
        Weight for foot-floor penetration / contact penalty.
    warmup_epochs:
        Number of epochs over which all weights are linearly ramped.
    max_flexion_deg:
        Maximum allowed interior joint angle.
    """

    def __init__(
        self,
        parents: List[int],
        symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
        foot_indices: Optional[List[int]] = None,
        bone_weight: float = 0.05,
        joint_limit_weight: float = 0.01,
        symmetry_weight: float = 0.02,
        floor_weight: float = 0.02,
        warmup_epochs: int = 0,
        max_flexion_deg: float = 160.0,
    ) -> None:
        super().__init__()
        self.parents = list(parents)
        self.symmetry_pairs = list(symmetry_pairs or [])
        self.foot_indices = list(foot_indices or [])
        self.bone_weight = bone_weight
        self.joint_limit_weight = joint_limit_weight
        self.symmetry_weight = symmetry_weight
        self.floor_weight = floor_weight
        self.warmup_epochs = warmup_epochs
        self.max_flexion_deg = max_flexion_deg

    def _ramp(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    def bone_length_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L2 loss between predicted and target bone lengths."""
        pred_bones, children = _bone_vectors(pred, self.parents)
        target_bones, _ = _bone_vectors(target, self.parents)
        pred_lengths = pred_bones.norm(dim=-1)
        target_lengths = target_bones.norm(dim=-1)
        return F.mse_loss(pred_lengths, target_lengths)

    def joint_limit_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Soft penalty on interior angles exceeding max_flexion_deg."""
        if pred.dim() == 4:  # (B, T, J, 3)
            B, T, J, _ = pred.shape
            pred = pred.reshape(B * T, J, 3)
        elif pred.dim() == 3:
            J = pred.shape[-2]
        else:
            raise ValueError(f"Unexpected pred shape {pred.shape}")

        cos_max = torch.cos(torch.deg2rad(torch.tensor(self.max_flexion_deg, device=pred.device)))
        losses = []
        for j, p in enumerate(self.parents):
            if p < 0:
                continue
            children = [c for c, par in enumerate(self.parents) if par == j]
            for c in children:
                b_in = F.normalize(pred[:, j, :] - pred[:, p, :], dim=-1)
                b_out = F.normalize(pred[:, c, :] - pred[:, j, :], dim=-1)
                cos = (b_in * b_out).sum(dim=-1)
                losses.append(torch.relu(cos_max - cos))
        if not losses:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()

    def symmetry_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalty on length difference between left/right symmetric bones."""
        if not self.symmetry_pairs:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Build bone lengths for each joint that has a parent.
        lengths = torch.stack(
            [torch.norm(pred[..., child, :] - pred[..., parent, :], dim=-1)
             for child, parent in enumerate(self.parents) if parent >= 0],
            dim=-1,
        )
        # Map child index -> position in lengths tensor.
        valid_children = [child for child, parent in enumerate(self.parents) if parent >= 0]
        child_to_idx = {child: i for i, child in enumerate(valid_children)}

        diffs = []
        for left, right in self.symmetry_pairs:
            if left in child_to_idx and right in child_to_idx:
                diffs.append(
                    (lengths[..., child_to_idx[left]] - lengths[..., child_to_idx[right]]) ** 2
                )
        if not diffs:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(diffs).mean()

    def floor_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalty on feet below floor (y=0) and high foot velocity at floor."""
        if not self.foot_indices:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        feet = pred[..., self.foot_indices, :]  # (B, T, n_feet, 3) or (B, n_feet, 3)
        if feet.dim() == 3:
            # Single-frame input; only floor penetration.
            return torch.relu(-feet[..., 1]).mean()
        # Multi-frame: penalize penetration and velocity when near floor.
        penetration = torch.relu(-feet[..., 1]).mean()
        velocity = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1).mean()
        return penetration + 0.1 * velocity

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Compute composite physical loss.

        Args:
            pred: ``(B, T, J, 3)`` or ``(B, J, 3)`` predicted 3-D pose.
            target: optional ground-truth pose, only used for the bone-length term.
            epoch: current epoch for weight ramp-up.

        Returns:
            Scalar loss tensor.
        """
        ramp = self._ramp(epoch)
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if self.bone_weight > 0.0 and target is not None:
            loss = loss + ramp * self.bone_weight * self.bone_length_loss(pred, target)

        if self.joint_limit_weight > 0.0:
            loss = loss + ramp * self.joint_limit_weight * self.joint_limit_loss(pred)

        if self.symmetry_weight > 0.0 and self.symmetry_pairs:
            loss = loss + ramp * self.symmetry_weight * self.symmetry_loss(pred)

        if self.floor_weight > 0.0 and self.foot_indices:
            loss = loss + ramp * self.floor_weight * self.floor_loss(pred)

        return loss
