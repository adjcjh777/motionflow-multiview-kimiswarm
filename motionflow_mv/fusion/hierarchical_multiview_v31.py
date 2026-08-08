"""v31: Geometry-Attention Refinement for the hierarchical multi-view encoder.

Builds on the v30 stable hierarchical encoder but biases each cross-view
attention scale with geometry-derived signals (epipolar distance + ray
intersection quality).  The block remains identity at initialization and keeps
v30's stochastic depth / gated residual hardening.

The intended integration point is the same as v30: after the per-view feature
tokens have been formed and before the spatio-temporal transformer.  Unlike v30,
this module also expects the camera parameters and 2-D observations so it can
compute geometry-aware attention biases on the fly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    compute_rays,
    ray_intersection_logit,
)


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
    return {f"joint_{i}": [i] for i in range(j)}


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    """Flatten batch and time dims: (B, T, ...) -> (B*T, ...)."""
    return x.reshape(-1, *x.shape[2:])


class _GeometryBiasedCrossViewAttentionBlock(nn.Module):
    """Cross-view attention block with optional geometry bias and stochastic depth.

    Wraps ``nn.MultiheadAttention`` directly so we can add a learned geometry
    bias to the attention scores before softmax.  The module keeps the same
    pre-norm + residual structure as v30, and is identity at init when the
    output projection is zeroed.
    """

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
        self.n_heads = n_heads
        self.stochastic_depth_prob = stochastic_depth_prob

        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

        # Zero the output projection so the residual path vanishes at init.
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        # Feed-forward network (same ratio as TransformerEncoderLayer default).
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d)

        # Stack ``n_layers`` identical layers by wrapping sub-modules in a list.
        if n_layers > 1:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(d),
                            "attn": nn.MultiheadAttention(
                                embed_dim=d,
                                num_heads=n_heads,
                                dropout=dropout,
                                batch_first=True,
                            ),
                            "out_proj": nn.Linear(d, d),
                            "ffn": nn.Sequential(
                                nn.Linear(d, d * 2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d * 2, d),
                                nn.Dropout(dropout),
                            ),
                            "ffn_norm": nn.LayerNorm(d),
                        }
                    )
                    for _ in range(n_layers - 1)
                ]
            )
            for layer in self.layers:
                for p in layer["out_proj"].parameters():
                    nn.init.zeros_(p)
        else:
            self.layers = nn.ModuleList()

        # Geometry bias gate: initialised to -3 -> sigmoid ~0.05, so geometry
        # contributes weakly at first and grows as training progresses.
        self.geometry_gate = nn.Parameter(torch.tensor(-3.0))

    def _attn_forward(
        self,
        x: torch.Tensor,
        geometry_bias: Optional[torch.Tensor],
        attn_module: nn.MultiheadAttention,
    ) -> torch.Tensor:
        """Single attention + FFN sub-layer with optional geometry bias.

        Args:
            x: (N, V, d) where N is the flattened batch.
            geometry_bias: optional (N, V, V) additive attention logit bias.
        """
        # content-only scores are computed by MultiheadAttention; we add the
        # geometry bias by passing it as attn_mask.  PyTorch's
        # MultiheadAttention adds attn_mask to the scores before softmax.
        if geometry_bias is not None:
            gate = torch.sigmoid(self.geometry_gate)
            attn_mask = gate * geometry_bias
            # PyTorch expects attn_mask of shape (N*num_heads, V, V).
            attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
        else:
            attn_mask = None
        out, _ = attn_module(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out

    def forward(
        self,
        x: torch.Tensor,
        geometry_bias: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            x: (B, V, d) tokens across views.
            geometry_bias: optional (B, V, V) attention logit bias.
            view_mask: optional (B, V) bool; True = keep view.
        Returns:
            out: (B, V, d).
        """
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            return x

        if view_mask is not None:
            # Zero out masked views so they cannot attend or be attended to.
            x = x * view_mask[:, :, None].float()

        # First layer.
        h = self.norm(x)
        out = self._attn_forward(h, geometry_bias, self.attn)
        out = x + self.out_proj(out)
        out = out + self.ffn(self.ffn_norm(out))

        # Additional stacked layers.
        for layer in self.layers:
            h = layer["norm"](out)
            h = self._attn_forward(h, geometry_bias, layer["attn"])
            out = out + layer["out_proj"](h)
            out = out + layer["ffn"](layer["ffn_norm"](out))

        if view_mask is not None:
            out = out * view_mask[:, :, None].float()

        return out


class HierarchicalViewEncoderV31(nn.Module):
    """Geometry-biased stable multi-scale view encoder.

    Combines v30's hardening (LayerNorm, stochastic depth, gated residual,
    zeroed output projections) with v25-style geometry-aware cross-view
    attention.  The geometry biases are computed once from the input cameras
    and 2-D points and are shared across the joint / part / body scales.

    If ``points_2d`` / cameras are omitted, the block falls back to v30
    content-only behaviour, which is useful for smoke tests and for warm-start
    from a v30 checkpoint.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        dropout: float = 0.1,
        n_part_layers: int = 1,
        stochastic_depth_prob: float = 0.0,
        n_geometry_layers: int = 1,
        use_ray_attention: bool = False,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_geometry_layers = n_geometry_layers
        self.use_ray_attention = use_ray_attention

        # Optional ray/camera embedding branch.  Zero-initialised so it is a no-op.
        if self.use_ray_attention:
            self.ray_proj = nn.Sequential(
                nn.Linear(6, d),
                nn.ReLU(),
                nn.Linear(d, d),
            )
            nn.init.zeros_(self.ray_proj[-1].weight)
            nn.init.zeros_(self.ray_proj[-1].bias)
            self.ray_gate = nn.Parameter(torch.zeros(1))

        # Per-scale attention blocks.
        self.joint_block = _GeometryBiasedCrossViewAttentionBlock(
            d, n_heads, n_layers=n_geometry_layers, dropout=dropout, stochastic_depth_prob=stochastic_depth_prob
        )
        self.part_block = _GeometryBiasedCrossViewAttentionBlock(
            d,
            max(1, n_heads // 2),
            n_layers=max(1, n_part_layers),
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.body_block = _GeometryBiasedCrossViewAttentionBlock(
            d,
            max(1, n_heads // 2),
            n_layers=n_geometry_layers,
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

        # Geometry temperature parameters for ray-intersection logit.
        self.sigma_d = nn.Parameter(torch.tensor(0.5))
        self.sigma_a = nn.Parameter(torch.tensor(0.5))

    def _compute_geometry_bias(
        self,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-joint geometry bias (B, T, V, V, J) from cameras + points."""
        B, T, V, J = points_2d.shape[:4]

        # Compute epipolar distances.  compute_epipolar_distance expects
        # (B, V, 3, 3) and (B, V, J, 2).  Flatten B*T to a single batch dim.
        K_flat = _flatten_bt(K)  # (B*T, V, 3, 3)
        R_flat = _flatten_bt(R)  # (B*T, V, 3, 3)
        t_flat = _flatten_bt(t)  # (B*T, V, 3)
        pts_flat = _flatten_bt(points_2d)  # (B*T, V, J, 2)

        epi_dist = compute_epipolar_distance(K_flat, R_flat, t_flat, pts_flat)
        # epi_dist is (B*T, V, V, J); reshape back.
        epi_dist = epi_dist.reshape(B, T, V, V, J)

        # Compute ray-intersection logit.
        centre, direction = compute_rays(points_2d, K, R, t)
        ray_logit = ray_intersection_logit(centre, direction, self.sigma_d, self.sigma_a)
        # ray_logit is (B, T, V, V, J)

        # Combine into a single additive bias.  Lower epipolar distance -> higher
        # attention, higher ray intersection logit -> higher attention.
        geometry_bias = -epi_dist + ray_logit
        return geometry_bias

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        points_2d: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            view_mask: optional (B, T, V) bool.
            points_2d: optional (B, T, V, J, 2).  Required for geometry bias.
            K: optional (B, T, V, 3, 3) or (B, V, 3, 3).  Required for geometry bias.
            R: optional (B, T, V, 3, 3) or (B, V, 3, 3).  Required for geometry bias.
            t: optional (B, T, V, 3) or (B, V, 3).  Required for geometry bias.
        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        part_groups = _part_groups_for_joints(J)
        group_indices = list(part_groups.values())

        # Compute geometry bias once and reuse across scales.
        geometry_bias = None
        if points_2d is not None and K is not None and R is not None and t is not None:
            # Broadcast single-time cameras to T if necessary.
            if K.dim() == 4:
                K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
            if R.dim() == 4:
                R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
            if t.dim() == 3:
                t = t.unsqueeze(1).expand(-1, T, -1, -1)
            geometry_bias = self._compute_geometry_bias(points_2d, K, R, t)

        # Optional ray-conditioned token bias.
        if self.use_ray_attention and points_2d is not None and K is not None and R is not None and t is not None:
            centre, direction = compute_rays(points_2d, K, R, t)
            ray_input = torch.cat([centre, direction], dim=-1)  # (B, T, V, J, 6)
            ray_emb = self.ray_proj(ray_input) * self.ray_gate  # (B, T, V, J, d)
            tokens = tokens + ray_emb

        # Joint scale: per-joint cross-view attention with geometry bias.
        joint_in = tokens.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        joint_bias = None
        if geometry_bias is not None:
            # (B, T, V, V, J) -> (B*T*J, V, V)
            joint_bias = geometry_bias.permute(0, 1, 4, 2, 3).reshape(B * T * J, V, V)
        view_mask_flat_joint = None
        if view_mask is not None:
            # (B, T, V) -> (B*T*J, V)
            view_mask_flat_joint = view_mask.unsqueeze(2).expand(-1, -1, J, -1).reshape(B * T * J, V)
        joint_out = self.joint_block(joint_in, geometry_bias=joint_bias, view_mask=view_mask_flat_joint)
        joint_out = joint_out.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)
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

        view_mask_flat_part = None
        if view_mask is not None:
            # (B, T, V) -> (B*T*Pn, V)
            view_mask_flat_part = view_mask.reshape(B * T, V).unsqueeze(1).expand(-1, Pn, -1).reshape(B * T * Pn, V)

        # Geometry bias at the part scale: average per-joint bias over each part.
        part_bias = None
        if geometry_bias is not None:
            part_bias_list = []
            for indices in group_indices:
                valid_indices = [i for i in indices if 0 <= i < J]
                if not valid_indices:
                    continue
                # geometry_bias: (B, T, V, V, J) -> average over J' in part.
                part_bias_list.append(geometry_bias[..., valid_indices].mean(dim=-1, keepdim=True))
            part_bias = torch.cat(part_bias_list, dim=-1)  # (B, T, V, V, Pn)
            part_bias = part_bias.permute(0, 1, 4, 2, 3).reshape(B * T * Pn, V, V)

        part_out = self.part_block(part_in, geometry_bias=part_bias, view_mask=view_mask_flat_part)
        part_out = part_out.view(B, T, Pn, V, d).permute(0, 1, 3, 2, 4)
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
        body_bias = None
        if geometry_bias is not None:
            # Body-scale bias averages over joints.
            body_bias = geometry_bias.mean(dim=-1)  # (B, T, V, V)
            body_bias = body_bias.reshape(B * T, V, V)
        body_view_mask = None
        if view_mask is not None:
            body_view_mask = view_mask.reshape(B * T, V)
        body_out = self.body_block(body_in, geometry_bias=body_bias, view_mask=body_view_mask)
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
