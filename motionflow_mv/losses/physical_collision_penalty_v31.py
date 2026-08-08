"""v31: Self-collision / self-penetration penalty for 3-D human pose.

Treats each bone as a capsule (line segment with radius) and penalises
non-adjacent bone pairs whose capsules intersect or come closer than a
safety margin.  The loss is differentiable and operates on the predicted
3-D pose ``(B, T, J, 3)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _segment_segment_distance(p1: torch.Tensor, p2: torch.Tensor,
                              q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Minimum Euclidean distance between two 3-D line segments.

    Args:
        p1, p2: endpoints of segment P, each ``(..., 3)``.
        q1, q2: endpoints of segment Q, each ``(..., 3)``.

    Returns:
        Scalar or shape-matching tensor of distances.
    """
    u = p2 - p1  # (..., 3)
    v = q2 - q1
    w0 = p1 - q1

    a = (u * u).sum(dim=-1)  # (...,)
    b = (u * v).sum(dim=-1)
    c = (v * v).sum(dim=-1)
    d = (u * w0).sum(dim=-1)
    e = (v * w0).sum(dim=-1)

    denom = a * c - b * b
    eps = 1e-8

    # Handle parallel or near-parallel segments by clamping denom.
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom) * eps, denom)

    # Closest points on infinite lines.
    s0 = (b * e - c * d) / denom
    t0 = (a * e - b * d) / denom

    # Clamp to segments.
    s = torch.clamp(s0, 0.0, 1.0)
    t = torch.clamp(t0, 0.0, 1.0)

    # Reproject when the unconstrained closest point lies outside the segment.
    # This is a cheap, robust approximation: it clamps each coordinate
    # independently, which is exact when the closest region is a vertex.
    pc = p1 + s.unsqueeze(-1) * u
    qc = q1 + t.unsqueeze(-1) * v
    dist = (pc - qc).norm(dim=-1)
    return dist


def _build_non_adjacent_pairs(parents: List[int]) -> List[Tuple[int, int]]:
    """Return bone indices (child index) of non-adjacent bones.

    Two bones are considered adjacent if they share the root joint, share the
    child joint, or one is the parent of the other.
    """
    # bones[i] = (parent, child)
    bones = [(p, c) for c, p in enumerate(parents) if p >= 0]
    n_bones = len(bones)
    pairs: List[Tuple[int, int]] = []
    for i in range(n_bones):
        for j in range(i + 1, n_bones):
            a = set(bones[i])
            b = set(bones[j])
            if a & b:
                continue
            pairs.append((i, j))
    return pairs


class PhysicalCollisionPenaltyV31(nn.Module):
    """Training-time self-collision penalty over pose sequences.

    Parameters
    ----------
    parents:
        Parent index list defining the skeleton bones.
    loss_weight:
        Overall multiplier for the collision penalty.
    bone_radius:
        Capsule radius around each bone.  Either a scalar or a tensor of
        length ``len(parents)``.
    margin:
        Minimum allowed distance between non-adjacent capsule surfaces in
        addition to their radii.
    warmup_epochs:
        Number of epochs over which the loss linearly ramps from 0 to 1.
    """

    def __init__(
        self,
        parents: List[int],
        loss_weight: float = 0.001,
        bone_radius: float | torch.Tensor = 0.07,
        margin: float = 0.05,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.parents = parents
        self.loss_weight = loss_weight
        self.margin = margin
        self.warmup_epochs = max(0, warmup_epochs)
        self.current_epoch = 0

        # Convert bone_radius to a list/tensor and store as buffer.
        if isinstance(bone_radius, float):
            radii = torch.full((len(parents),), bone_radius)
        else:
            radii = torch.as_tensor(bone_radius, dtype=torch.float32)
        self.register_buffer("bone_radii", radii)

        # Build list of non-adjacent bone index pairs.
        pairs = _build_non_adjacent_pairs(parents)
        if pairs:
            pair_tensor = torch.tensor(pairs, dtype=torch.long)  # (n_pairs, 2)
        else:
            pair_tensor = torch.empty((0, 2), dtype=torch.long)
        self.register_buffer("bone_pairs", pair_tensor)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Args:
            X: ``(B, T, J, 3)`` predicted poses.

        Returns:
            total: scalar loss.
            terms: dictionary with ``collision`` key.
        """
        if self.bone_pairs.numel() == 0:
            loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
            return loss, {"collision": loss}

        # Build bone endpoints.  Bone i goes from parent to child using the
        # stored parent list.  We need a mapping from bone index to child joint.
        bones = [(p, c) for c, p in enumerate(self.parents) if p >= 0]
        parent_indices = torch.tensor([p for p, _ in bones], device=X.device, dtype=torch.long)
        child_indices = torch.tensor([c for _, c in bones], device=X.device, dtype=torch.long)

        # Gather endpoints: (B, T, n_bones, 3)
        p_start = X[..., parent_indices, :]  # (B, T, n_bones, 3)
        p_end = X[..., child_indices, :]

        # Select the two bones in each pair.
        idx_a = self.bone_pairs[:, 0]
        idx_b = self.bone_pairs[:, 1]

        a_start = p_start[..., idx_a, :]  # (B, T, n_pairs, 3)
        a_end = p_end[..., idx_a, :]
        b_start = p_start[..., idx_b, :]
        b_end = p_end[..., idx_b, :]

        # Minimum distance between the two bone segments.
        dist = _segment_segment_distance(a_start, a_end, b_start, b_end)

        # Threshold = sum of radii + margin.
        radii_a = self.bone_radii[idx_a].to(X.dtype)
        radii_b = self.bone_radii[idx_b].to(X.dtype)
        threshold = radii_a + radii_b + self.margin

        # Repulsion loss: positive only when capsules overlap or are too close.
        penetration = (threshold - dist).clamp(min=0.0)
        collision_loss = (penetration ** 2).mean()

        # Warmup scaling.
        scale = 1.0
        if self.warmup_epochs > 0:
            scale = min(1.0, self.current_epoch / self.warmup_epochs)

        total = scale * self.loss_weight * collision_loss
        return total, {"collision": total.detach(), "collision_raw": collision_loss.detach()}
