"""v30: Stable Hierarchical Multi-View Encoder.

Builds on v29's multi-scale idea but hardens it against overfitting:

* Dataset-aware part groups (H36M 17-joint, MPI 28-joint, generic fallback).
* Cross-scale residual fusion with gated scale weights.
* Stochastic depth / dropout in cross-view attention.
* LayerNorm + scaled residual paths for stable training at larger capacity.
* Identity-at-init is preserved via zeroed output projections and a learned
  residual gate initialised near zero.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Skeleton part definitions (zero-indexed).
# ---------------------------------------------------------------------------

H36M_17_PART_GROUPS = {
    "head": [10, 11, 12, 13],
    "torso": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "l_arm": [11, 12],
    "r_arm": [13, 14],
    "l_leg": [3, 4, 5],
    "r_leg": [6, 7, 8],
}

MPI_28_PART_GROUPS = {
    "head": [0, 1, 2, 3, 4, 5, 6],
    "torso": [7, 8, 9, 10, 11, 12],
    "l_arm": [13, 14, 15, 16, 17],
    "r_arm": [18, 19, 20, 21, 22],
    "l_leg": [23, 24, 25],
    "r_leg": [26, 27],
}


def _part_groups_for_joints(j: int) -> Dict[str, List[int]]:
    if j == 17:
        return H36M_17_PART_GROUPS
    if j == 28:
        return MPI_28_PART_GROUPS
    # Fallback: every joint is its own group.
    return {f"joint_{i}": [i] for i in range(j)}


class _CrossViewAttentionBlock(nn.Module):
    """Lightweight cross-view transformer block with optional stochastic depth."""

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.stochastic_depth_prob = stochastic_depth_prob
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, V, d) tokens across views.
        Returns:
            out: (B, V, d).
        """
        # Stochastic depth: randomly drop the block during training.
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            return x
        out = self.encoder(x)
        out = self.norm(out)
        return x + out  # residual


class HierarchicalViewEncoderV30(nn.Module):
    """Stable multi-scale view encoder.

    Combines joint-level, part-level, and body-level cross-view attention with
    learned scale weights and a small residual MLP.  The block is identity at
    init: the per-scale output projections are zeroed and the final residual
    gate is initialised near zero.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        dropout: float = 0.1,
        n_part_layers: int = 1,
        stochastic_depth_prob: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views

        # Per-scale attention blocks.
        self.joint_block = _CrossViewAttentionBlock(
            d, n_heads, n_layers=1, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob
        )
        self.part_block = _CrossViewAttentionBlock(
            d,
            max(1, n_heads // 2),
            n_layers=n_part_layers,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.body_block = _CrossViewAttentionBlock(
            d,
            max(1, n_heads // 2),
            n_layers=1,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        # Output projections for each scale (zeroed at init).
        self.joint_proj = nn.Linear(d, d)
        self.part_proj = nn.Linear(d, d)
        self.body_proj = nn.Linear(d, d)
        for proj in (self.joint_proj, self.part_proj, self.body_proj):
            for p in proj.parameters():
                nn.init.zeros_(p)

        # Learned scale weights (softmax over 3 scales), initialised near equal.
        self.scale_logits = nn.Parameter(torch.zeros(3))

        # Gated residual: at init the gate is ~0 so the block is identity.
        self.residual_gate = nn.Parameter(torch.tensor(-6.0))

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            view_mask: optional (B, T, V) bool.
        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        part_groups = _part_groups_for_joints(J)
        group_indices = list(part_groups.values())

        # Joint scale: per-joint cross-view attention.
        joint_in = tokens.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        joint_out = self.joint_block(joint_in)
        joint_out = joint_out.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, J, d)
        joint_out = self.joint_proj(joint_out)

        # Part scale: aggregate within each part, then cross-view attention.
        part_tokens_list = []
        for indices in group_indices:
            valid_indices = [i for i in indices if 0 <= i < J]
            if not valid_indices:
                continue
            part_tok = tokens[:, :, :, valid_indices, :].mean(dim=3, keepdim=True)
            part_tokens_list.append(part_tok)
        part_scale_tokens = torch.cat(part_tokens_list, dim=3)  # (B, T, V, P, d)
        Pn = part_scale_tokens.shape[3]
        part_in = part_scale_tokens.permute(0, 1, 3, 2, 4).reshape(B * T * Pn, V, d)
        part_out = self.part_block(part_in)
        part_out = part_out.view(B, T, Pn, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, P, d)
        part_out = self.part_proj(part_out)

        # Map each part back to its constituent joints.
        part_to_joint = torch.zeros(B, T, V, J, d, device=tokens.device, dtype=tokens.dtype)
        part_idx = 0
        for indices in group_indices:
            valid_indices = [i for i in indices if 0 <= i < J]
            if not valid_indices:
                continue
            for j in valid_indices:
                part_to_joint[:, :, :, j, :] += part_out[:, :, :, part_idx, :]
            part_idx += 1

        # Body scale: single global token per view.
        body_token = tokens.mean(dim=3, keepdim=True)  # (B, T, V, 1, d)
        body_in = body_token.squeeze(3).view(B * T, V, d)
        body_out = self.body_block(body_in)
        body_out = body_out.view(B, T, V, d)
        body_out = self.body_proj(body_out[:, :, :, None, :]).expand(-1, -1, -1, J, -1)

        # Apply view mask if provided.
        if view_mask is not None:
            mask = view_mask[:, :, :, None, None]
            joint_out = joint_out * mask.float()
            part_to_joint = part_to_joint * mask.float()
            body_out = body_out * mask.float()

        # Combine scales with learned weights.
        scale_weights = F.softmax(self.scale_logits, dim=0)
        multi_scale = (
            scale_weights[0] * joint_out
            + scale_weights[1] * part_to_joint
            + scale_weights[2] * body_out
        )

        # Gated residual addition.
        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * multi_scale
