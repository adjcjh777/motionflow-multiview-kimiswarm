"""v28: Physical-space alignment for multi-view 3D human pose.

A lightweight learned refiner that enforces gravity/floor and bone-length
 temporal consistency on the final 3D pose.  Initialised as an no-op so it
can be safely enabled without changing existing checkpoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhysicalSpaceAlignmentV28(nn.Module):
    """Learned physical-space alignment refiner.

    Parameters
    ----------
    j:
        Number of joints.
    hidden:
        Hidden dimension of the residual MLP.
    """

    def __init__(self, j: int, hidden: int = 64):
        super().__init__()
        self.j = j
        self.gravity_dir = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]), requires_grad=False)

        self.refiner = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        # Identity at init: residual is zero.
        for p in self.refiner[-1].parameters():
            nn.init.zeros_(p)

        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        X: torch.Tensor,
        gravity_dir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine pose with physical-space constraints.

        Args:
            X: (B, T, J, 3) predicted 3D joints.
            gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).

        Returns:
            X_aligned: (B, T, J, 3) refined 3D joints.
        """
        if gravity_dir is None:
            gravity_dir = self.gravity_dir

        B, T, J, _ = X.shape
        # Broadcast gravity direction.
        g = gravity_dir.to(X.device, X.dtype)
        g = g.view(1, 1, 1, 3).expand(B, T, J, -1)

        feat = torch.cat([X, g], dim=-1)
        residual = self.refiner(feat)
        return X + self.residual_scale * residual


def floor_loss(
    X: torch.Tensor,
    floor_height: torch.Tensor,
    foot_joint_indices: list[int],
    gravity_dir: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft hinge loss penalising foot joints below the floor plane.

    Args:
        X: (B, T, J, 3) 3D joints.
        floor_height: scalar or (B, T) tensor of floor heights along gravity.
        foot_joint_indices: list of foot joint indices.
        gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).

    Returns:
        Scalar loss.
    """
    if gravity_dir is None:
        gravity_dir = torch.tensor([0.0, 1.0, 0.0], device=X.device, dtype=X.dtype)
    g = gravity_dir / (gravity_dir.norm() + 1e-8)
    # Project joints onto gravity axis.
    h = torch.einsum("btjc,c->btj", X, g)
    feet = h[:, :, foot_joint_indices]
    violation = (floor_height - feet).clamp(min=0.0)
    return violation.mean()


def bone_temporal_loss(
    X: torch.Tensor,
    parents: list[int],
) -> torch.Tensor:
    """Temporal consistency of bone lengths.

    Args:
        X: (B, T, J, 3) 3D joints.
        parents: list of parent indices.

    Returns:
        Scalar loss: mean squared change in bone length over time.
    """
    if X.shape[1] < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    bone_vecs = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bone = X[..., child, :] - X[..., parent, :]
        bone_vecs.append(bone)

    if not bone_vecs:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    bones = torch.stack(bone_vecs, dim=-2)  # (B, T, n_bones, 3)
    lengths = bones.norm(dim=-1)  # (B, T, n_bones)
    diff = lengths[:, 1:] - lengths[:, :-1]
    return diff.pow(2).mean()
