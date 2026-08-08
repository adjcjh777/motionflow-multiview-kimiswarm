"""v29: Self-Evolving Hierarchical Multi-View Fusion (SEH-MV).

Three lightweight, flag-gated additions to the v25 backbone:

1. ``HierarchicalViewEncoderV29`` – fuses views at joint / part / body scales.
2. ``TestTimeSelfEvolutionV29`` – iterative geometric self-consistency with an
   optional physical-space alignment step.
3. ``PhysicalSpaceTemporalLossV29`` – training-time physical prior over sequences.

All blocks are identity-at-init when possible and accept ``view_mask`` for
variable-view training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.physical_space_alignment_v28 import (
    PhysicalSpaceAlignmentV28,
    floor_loss as floor_loss_fn,
)
from motionflow_mv.fusion.test_time_self_evolution_v27 import TestTimeSelfEvolutionV27


# ---------------------------------------------------------------------------
# Skeleton part definitions (zero-indexed, H36M 17-joint order).
# ---------------------------------------------------------------------------

H36M_PART_GROUPS = {
    "head": [10, 11, 12, 13],
    "torso": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "l_arm": [11, 12],
    "r_arm": [13, 14],
    "l_leg": [3, 4, 5],
    "r_leg": [6, 7, 8],
}

MPI_PART_GROUPS = {
    "head": [0, 1, 2, 3, 4, 5, 6],
    "torso": [7, 8, 9, 10, 11, 12],
    "l_arm": [13, 14, 15, 16, 17],
    "r_arm": [18, 19, 20, 21, 22],
    "l_leg": [23, 24, 25],
    "r_leg": [26, 27],
}


def _part_group_for_joints(j: int) -> Dict[str, List[int]]:
    if j == 17:
        return H36M_PART_GROUPS
    if j == 28:
        return MPI_PART_GROUPS
    # Fallback: every joint is its own group.
    return {f"joint_{i}": [i] for i in range(j)}


class HierarchicalViewEncoderV29(nn.Module):
    """Multi-scale view encoder: joint / part / body.

    Each scale runs a lightweight cross-view attention block.  Outputs are
    combined with learned scale weights.  The body scale provides a global
    pose token; the part scale provides limb/torso redundancy for few views;
    the joint scale preserves fine detail for many views.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        dropout: float = 0.1,
        n_part_layers: int = 1,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views

        # Three scale-specific projections.
        self.joint_proj = nn.Linear(d, d)
        self.part_proj = nn.Linear(d, d)
        self.body_proj = nn.Linear(d, d)

        # Cross-view attention for each scale.  We reuse a simple transformer
        # encoder layer for permutation-equivariant view mixing.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.joint_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer_part = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=max(1, n_heads // 2),
            dim_feedforward=d * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.part_attn = nn.TransformerEncoder(encoder_layer_part, num_layers=n_part_layers)

        encoder_layer_body = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=max(1, n_heads // 2),
            dim_feedforward=d * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.body_attn = nn.TransformerEncoder(encoder_layer_body, num_layers=1)

        # Soft scale weights, initialised near equal.
        self.scale_logits = nn.Parameter(torch.zeros(3))

        # Identity at init: zero the output of the final projections so the
        # residual contribution vanishes at start of training.
        for proj in (self.joint_proj, self.part_proj, self.body_proj):
            for p in proj.parameters():
                nn.init.zeros_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            view_mask: optional (B, T, V) bool.

        Returns:
            refined: (B, T, V, J, d) residual-free at init.
        """
        B, T, V, J, d = tokens.shape

        # Determine part groups based on joint count.
        part_groups = _part_group_for_joints(J)
        group_indices = list(part_groups.values())

        # ---- Joint scale: per-joint cross-view attention -------------------
        # (B*T*J, V, d)
        joint_in = tokens.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        joint_mask = self._mask(view_mask, B, T, V, repeats=J)
        joint_out = self.joint_attn(joint_in, src_key_padding_mask=joint_mask)
        joint_out = joint_out.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, J, d)
        joint_out = self.joint_proj(joint_out)

        # ---- Part scale: aggregate within each part, then cross-view --------
        # Build part tokens by averaging joint tokens within each part group.
        part_tokens_list = []
        for indices in group_indices:
            valid_indices = [i for i in indices if 0 <= i < J]
            if not valid_indices:
                continue
            part_tok = tokens[:, :, :, valid_indices, :].mean(dim=3, keepdim=True)  # (B, T, V, 1, d)
            part_tokens_list.append(part_tok)
        # If a part group is empty (should not happen), skip it.
        part_scale_tokens = torch.cat(part_tokens_list, dim=3)  # (B, T, V, P, d)
        P = part_scale_tokens.shape[3]
        part_in = part_scale_tokens.permute(0, 1, 3, 2, 4).reshape(B * T * P, V, d)
        part_mask = self._mask(view_mask, B, T, V, repeats=P)
        part_out = self.part_attn(part_in, src_key_padding_mask=part_mask)
        part_out = part_out.view(B, T, P, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, P, d)
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

        # ---- Body scale: single global token per view -----------------------
        body_token = tokens.mean(dim=3, keepdim=True)  # (B, T, V, 1, d)
        body_in = body_token.squeeze(3)  # (B, T, V, d)
        body_in = body_in.view(B * T, V, d)
        body_mask = self._mask(view_mask, B, T, V, repeats=1)
        body_out = self.body_attn(body_in, src_key_padding_mask=body_mask)
        body_out = body_out.view(B, T, V, d)
        body_out = self.body_proj(body_out[:, :, :, None, :]).expand(-1, -1, -1, J, -1)

        # ---- Combine with learned scale weights ------------------------------
        scale_weights = F.softmax(self.scale_logits, dim=0)  # (3,)
        refined = (
            scale_weights[0] * joint_out
            + scale_weights[1] * part_to_joint
            + scale_weights[2] * body_out
        )
        return refined

    def _mask(
        self,
        view_mask: Optional[torch.Tensor],
        B: int,
        T: int,
        V: int,
        repeats: int = 1,
    ) -> Optional[torch.Tensor]:
        if view_mask is None:
            return None
        # view_mask: (B, T, V) -> repeat for each sub-entity (joint/part/body)
        # -> (B*T*repeats, V)
        mask = view_mask.reshape(B * T, V).bool()
        if repeats > 1:
            mask = mask.unsqueeze(1).repeat(1, repeats, 1).reshape(B * T * repeats, V)
        return ~mask


class TestTimeSelfEvolutionV29(nn.Module):
    """v27 TTE wrapped with an v28 physical-space alignment step.

    At inference, the model refines its prediction by iterating between
    reprojection-based view re-weighting and optional physical-space alignment.
    """

    def __init__(
        self,
        n_iters: int = 3,
        residual_thresh_mm: float = 0.5,
        sigma_reproj: float = 5.0,
        use_physical_space_alignment: bool = True,
        max_residual: float = 0.05,
        j: int = 17,
    ):
        super().__init__()
        self.tte = TestTimeSelfEvolutionV27(
            n_iters=n_iters,
            residual_thresh_mm=residual_thresh_mm,
            sigma_reproj=sigma_reproj,
        )
        self.use_physical_space_alignment = use_physical_space_alignment
        if use_physical_space_alignment:
            self.physical_space_alignment = PhysicalSpaceAlignmentV28(
                j=j, max_residual=max_residual
            )
        else:
            self.physical_space_alignment = None

    def forward(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        refined = self.tte(pred_3d, points_2d, K, R, t, view_mask=view_mask, confidence=confidence)
        if self.use_physical_space_alignment and self.physical_space_alignment is not None:
            refined = self.physical_space_alignment(refined)
        return refined


class PhysicalSpaceTemporalLossV29(nn.Module):
    """Training-time physical prior over sequences.

    Penalises foot-floor penetration, bone-length changes across time, and
    center-of-mass jitter.  All losses are normalised by batch size and are
    therefore O(1) in magnitude.
    """

    def __init__(
        self,
        floor_loss_weight: float = 0.01,
        bone_temporal_weight: float = 0.01,
        com_jitter_weight: float = 0.001,
        foot_joint_indices: Optional[List[int]] = None,
        parents: Optional[List[int]] = None,
    ):
        super().__init__()
        self.floor_loss_weight = floor_loss_weight
        self.bone_temporal_weight = bone_temporal_weight
        self.com_jitter_weight = com_jitter_weight
        self.foot_joint_indices = foot_joint_indices
        self.parents = parents

    def forward(
        self,
        X: torch.Tensor,
        parents: Optional[List[int]] = None,
        foot_joint_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Args:
            X: (B, T, J, 3).
            parents: optional list of parent indices for bone loss.
            foot_joint_indices: optional list of foot joint indices for floor loss.

        Returns:
            loss: scalar total loss.
            terms: dict of individual loss terms.
        """
        parents = parents if parents is not None else self.parents
        foot_joint_indices = foot_joint_indices if foot_joint_indices is not None else self.foot_joint_indices

        terms: Dict[str, torch.Tensor] = {}

        # Foot-floor penetration.
        floor = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if foot_joint_indices is not None and len(foot_joint_indices) > 0:
            floor = floor_loss_fn(X, floor_height=0.0, foot_joint_indices=foot_joint_indices)
        terms["floor"] = floor

        # Bone temporal smoothness.
        bone_temp = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if parents is not None and len(parents) > 0 and X.shape[1] > 1:
            bone_temp = self._bone_temporal_loss(X, parents)
        terms["bone_temporal"] = bone_temp

        # Center-of-mass jitter.
        com_jitter = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if X.shape[1] > 1:
            com = X.mean(dim=-2, keepdim=True)  # (B, T, 1, 3)
            com_jitter = (com[:, 1:] - com[:, :-1]).pow(2).mean()
        terms["com_jitter"] = com_jitter

        total = (
            self.floor_loss_weight * floor
            + self.bone_temporal_weight * bone_temp
            + self.com_jitter_weight * com_jitter
        )
        return total, terms

    @staticmethod
    def _bone_temporal_loss(X: torch.Tensor, parents: List[int]) -> torch.Tensor:
        bone_vecs = [X[..., c, :] - X[..., p, :] for c, p in enumerate(parents) if p >= 0]
        if not bone_vecs:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)
        bones = torch.stack(bone_vecs, dim=-2)  # (B, T, n_bones, 3)
        lengths = bones.norm(dim=-1)  # (B, T, n_bones)
        if lengths.shape[1] < 2:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)
        return (lengths[:, 1:] - lengths[:, :-1]).pow(2).mean()
