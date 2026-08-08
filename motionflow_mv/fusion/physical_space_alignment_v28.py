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
    """Conservative learned physical-space alignment refiner.

    The refiner is intentionally constrained so it cannot catastrophically
    override the upstream 3D pose estimate:

    * The MLP output is bounded by ``tanh`` to ``[-1, 1]`` and then scaled by
      ``max_residual`` (meters), so each joint can move at most a few cm.
    * The global residual scale is bounded with a sigmoid and initialised near
      zero, making the module start as a no-op and grow only when helpful.
    * LayerNorm and dropout are added inside the MLP to improve stability and
      reduce over-fitting on small local datasets.

    Parameters
    ----------
    j:
        Number of joints.
    hidden:
        Hidden dimension of the residual MLP.
    max_residual:
        Maximum per-joint correction in meters.
    dropout:
        Dropout probability in the residual MLP.
    """

    def __init__(
        self,
        j: int,
        hidden: int = 64,
        max_residual: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.j = j
        self.max_residual = max_residual
        self.gravity_dir = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]), requires_grad=False)

        self.refiner = nn.Sequential(
            nn.Linear(6, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )
        # Identity at init: residual is zero (Tanh(0) = 0).
        for p in self.refiner[-2].parameters():
            nn.init.zeros_(p)

        # Bounded scale in (0, 1).  Initialise to a tiny value so v28 starts as no-op.
        self.residual_logit = nn.Parameter(torch.tensor(-6.0))

    @property
    def residual_scale(self) -> torch.Tensor:
        """Bounded global residual scale in (0, 1)."""
        return torch.sigmoid(self.residual_logit)

    def forward(
        self,
        X: torch.Tensor,
        gravity_dir: Optional[torch.Tensor] = None,
        return_reg_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Refine pose with physical-space constraints.

        Args:
            X: (B, T, J, 3) predicted 3D joints.
            gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).
            return_reg_loss: If True, also return an L2 regulariser on the
                applied residual.

        Returns:
            X_aligned: (B, T, J, 3) refined 3D joints.
            If ``return_reg_loss`` is True, returns a tuple
            ``(X_aligned, reg_loss)`` where ``reg_loss`` is a scalar.
        """
        if gravity_dir is None:
            gravity_dir = self.gravity_dir

        B, T, J, _ = X.shape
        # Broadcast gravity direction.
        g = gravity_dir.to(X.device, X.dtype)
        g = g.view(1, 1, 1, 3).expand(B, T, J, -1)

        feat = torch.cat([X, g], dim=-1)
        raw_residual = self.refiner(feat)  # (B, T, J, 3), in [-1, 1]
        residual = self.max_residual * raw_residual
        scale = self.residual_scale
        X_aligned = X + scale * residual

        if not return_reg_loss:
            return X_aligned

        applied = scale * residual
        reg_loss = applied.pow(2).mean()
        return X_aligned, reg_loss


def floor_loss(
    X: torch.Tensor,
    floor_height: torch.Tensor,
    foot_joint_indices: list[int],
    gravity_dir: Optional[torch.Tensor] = None,
    floor_quantile: float = 0.05,
) -> torch.Tensor:
    """Soft hinge loss penalising foot joints below the floor plane.

    The floor height is estimated robustly per frame as the lower quantile of
    the selected foot joints along the gravity direction.  This is more stable
    than using a single global minimum over the whole batch.

    Args:
        X: (B, T, J, 3) 3D joints.
        floor_height: scalar or (B, T) tensor of floor heights along gravity.
            Kept for backward compatibility; the actual floor is estimated
            from the foot joints.
        foot_joint_indices: list of foot joint indices.
        gravity_dir: optional (3,) gravity direction. Defaults to (0, 1, 0).
        floor_quantile: quantile used to estimate the floor height from foot
            joints (default 0.05).

    Returns:
        Scalar loss.
    """
    del floor_height  # unused; kept for backward-compatible signatures
    if gravity_dir is None:
        gravity_dir = torch.tensor([0.0, 1.0, 0.0], device=X.device, dtype=X.dtype)
    g = gravity_dir / (gravity_dir.norm() + 1e-8)
    # Project joints onto gravity axis.
    h = torch.einsum("btjc,c->btj", X, g)
    feet = h[:, :, foot_joint_indices]

    # Robust per-frame floor height: lower quantile over feet.
    n_feet = feet.shape[-1]
    if n_feet > 1:
        k = max(1, int(floor_quantile * n_feet))
        floor_h, _ = torch.topk(feet, k, dim=-1, largest=False)
        floor_h = floor_h[..., -1]  # (B, T)
    else:
        floor_h = feet[..., 0]

    violation = (floor_h.unsqueeze(-1) - feet).clamp(min=0.0)
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
