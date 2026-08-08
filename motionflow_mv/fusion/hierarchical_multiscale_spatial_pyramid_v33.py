"""v33: Hierarchical multi-scale cross-view spatial pyramid.

Builds on the v31 geometry-biased cross-view attention but runs it at
multiple spatial scales over the joint dimension.  The block remains
identity at initialization and is compatible with variable-view training.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance
from motionflow_mv.fusion.hierarchical_multiview_v31 import (
    _GeometryBiasedCrossViewAttentionBlock,
)
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    compute_rays,
    ray_intersection_logit,
)


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    """Flatten batch and time dims: (B, T, ...) -> (B*T, ...)."""
    return x.reshape(-1, *x.shape[2:])


class HierarchicalMultiscaleCrossViewSpatialPyramidV33(nn.Module):
    """Multi-scale cross-view spatial pyramid.

    Args:
        d: token dimension.
        n_heads: number of attention heads per scale block.
        n_views: maximum number of padded views (used for mask bookkeeping only).
        scales: joint downsampling factors. Default (1, 2, 4).
        n_part_layers: number of layers inside each per-scale cross-view block.
        dropout: dropout rate inside cross-view attention.
        stochastic_depth_prob: stochastic depth probability.
        use_geometry_bias: whether to inject v31 epipolar/ray biases.
        use_adaptive_scale_fusion: whether to use per-token scale attention
            (True) or a fixed softmax weighting (False).
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        n_part_layers: int = 1,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.0,
        use_geometry_bias: bool = True,
        use_adaptive_scale_fusion: bool = True,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(int(s) for s in scales)
        self.use_geometry_bias = use_geometry_bias
        self.use_adaptive_scale_fusion = use_adaptive_scale_fusion

        # One cross-view attention block per scale.
        self.blocks = nn.ModuleList(
            [
                _GeometryBiasedCrossViewAttentionBlock(
                    d=d,
                    n_heads=n_heads,
                    n_layers=n_part_layers,
                    dropout=dropout,
                    stochastic_depth_prob=stochastic_depth_prob,
                )
                for _ in self.scales
            ]
        )

        # Fusion weights.
        if self.use_adaptive_scale_fusion:
            self.fusion_query = nn.Linear(d, d)
            nn.init.zeros_(self.fusion_query.weight)
            nn.init.zeros_(self.fusion_query.bias)
        else:
            self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

        # Output projections per scale, zeroed at init.
        self.out_projs = nn.ModuleList([nn.Linear(d, d) for _ in self.scales])
        for proj in self.out_projs:
            for p in proj.parameters():
                nn.init.zeros_(p)

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

        K_flat = _flatten_bt(K)  # (B*T, V, 3, 3)
        R_flat = _flatten_bt(R)  # (B*T, V, 3, 3)
        t_flat = _flatten_bt(t)  # (B*T, V, 3)
        pts_flat = _flatten_bt(points_2d)  # (B*T, V, J, 2)

        epi_dist = compute_epipolar_distance(K_flat, R_flat, t_flat, pts_flat)
        epi_dist = epi_dist.reshape(B, T, V, V, J)

        centre, direction = compute_rays(points_2d, K, R, t)
        ray_logit = ray_intersection_logit(centre, direction, self.sigma_d, self.sigma_a)

        return -epi_dist + ray_logit

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
            points_2d: optional (B, T, V, J, 2). Required for geometry bias.
            K: optional (B, T, V, 3, 3). Required for geometry bias.
            R: optional (B, T, V, 3, 3). Required for geometry bias.
            t: optional (B, T, V, 3). Required for geometry bias.
        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape

        # Compute full-resolution geometry bias once.
        geometry_bias = None
        if self.use_geometry_bias and points_2d is not None and K is not None and R is not None and t is not None:
            geometry_bias = self._compute_geometry_bias(points_2d, K, R, t)

        outputs = []
        for idx, s in enumerate(self.scales):
            J_s = max(1, J // s)

            # Downsample tokens along joint dimension: (B,T,V,J,d) -> (B,T,V,J_s,d).
            if s == 1:
                tok_s = tokens
            else:
                x = tokens.permute(0, 1, 2, 4, 3).reshape(B * T * V, d, J)
                x = F.adaptive_avg_pool1d(x, J_s)  # (B*T*V, d, J_s)
                x = x.view(B, T, V, d, J_s)
                tok_s = x.permute(0, 1, 2, 4, 3)  # (B, T, V, J_s, d)

            # Downsample geometry bias to the same joint resolution.
            gb_s: Optional[torch.Tensor] = None
            if geometry_bias is not None:
                # geometry_bias: (B, T, V, V, J) -> pool over J to J_s.
                gb = geometry_bias.permute(0, 1, 4, 2, 3).reshape(B * T * V * V, J)
                gb = F.adaptive_avg_pool1d(gb.unsqueeze(1), J_s).squeeze(1)  # (B*T*V*V, J_s)
                gb = gb.view(B, T, J_s, V, V).permute(0, 1, 3, 4, 2)  # (B, T, V, V, J_s)
                # (B, T, J_s, V, V) -> (B*T*J_s, V, V) additive attention bias.
                gb_s = gb.permute(0, 1, 4, 2, 3).reshape(B * T * J_s, V, V)

            # Flatten for cross-view attention: (B*T*J_s, V, d).
            x = tok_s.permute(0, 1, 3, 2, 4).reshape(B * T * J_s, V, d)

            # Flatten view mask.
            vm_s: Optional[torch.Tensor] = None
            if view_mask is not None:
                # view_mask: (B, T, V) -> (B*T*J_s, V)
                vm_s = view_mask.unsqueeze(2).expand(-1, -1, J_s, -1).reshape(B * T * J_s, V)

            out = self.blocks[idx](x, geometry_bias=gb_s, view_mask=vm_s)
            out = out.view(B, T, J_s, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, J_s, d)
            out = self.out_projs[idx](out)

            # Upsample back to full joint resolution.
            if s != 1:
                out = out.permute(0, 1, 2, 4, 3).reshape(B * T * V, d, J_s)
                out = F.interpolate(out, size=J, mode="linear", align_corners=False)
                out = out.view(B, T, V, d, J).permute(0, 1, 2, 4, 3)

            outputs.append(out)

        # Fuse multi-scale outputs.
        if self.use_adaptive_scale_fusion:
            # stack: (S, B, T, V, J, d)
            stack = torch.stack(outputs, dim=0)
            q = self.fusion_query(tokens)  # (B, T, V, J, d)
            logits = torch.einsum("btvjd,sbtvjd->btvjs", q, stack) / math.sqrt(self.d)
            weights = F.softmax(logits, dim=-1)  # (B, T, V, J, S)
            fused = torch.einsum("btvjs,sbtvjd->btvjd", weights, stack)
        else:
            w = F.softmax(self.scale_logits, dim=0)  # (S,)
            fused = sum(w[i] * outputs[i] for i in range(len(self.scales)))

        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * fused
