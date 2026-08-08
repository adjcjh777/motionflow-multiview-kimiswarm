"""v33 ray-conditioned cross-view attention.

This module makes cross-view attention explicitly geometry-aware by adding
per-joint ray embeddings directly to the query/key projections.  It is placed
after the v31 hierarchical encoder and is identity at init.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    compute_rays,
    ray_intersection_logit,
)
from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance


class _RayEmbedding(nn.Module):
    """Map per-joint ray features to the model dimension."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        # Input: camera centre (3) + direction (3) = 6.
        self.mlp = nn.Sequential(
            nn.Linear(6, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(
        self,
        centre: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            centre: (B, T, V, 3)
            direction: (B, T, V, J, 3)
        Returns:
            emb: (B, T, V, J, d)
        """
        # Broadcast centre to joints.
        centre = centre[:, :, :, None, :].expand(-1, -1, -1, direction.shape[-2], -1)
        x = torch.cat([direction, centre], dim=-1)  # (B, T, V, J, 6)
        return self.mlp(x)


class _RayConditionedCrossViewAttentionLayer(nn.Module):
    """Single ray-conditioned cross-view attention layer."""

    def __init__(
        self,
        d: int,
        n_heads: int,
        dropout: float = 0.1,
        use_ray_bias: bool = True,
        residual_gate_init: float = -6.0,
    ):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.use_ray_bias = use_ray_bias

        # Content projections.
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)

        # Ray projections added to Q/K.
        self.q_ray_proj = nn.Linear(d, d, bias=False)
        self.k_ray_proj = nn.Linear(d, d, bias=False)

        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)

        # Identity-at-init gate.
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

        # Optional learnable geometry temperature for ray intersection logit.
        if self.use_ray_bias:
            self.sigma_d = nn.Parameter(torch.tensor(0.5))
            self.sigma_a = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        tokens: torch.Tensor,
        ray_emb: torch.Tensor,
        ray_bias: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d)
            ray_emb: (B, T, V, J, d)
            ray_bias: optional (B, T, V, V, J) additive bias
            view_mask: optional (B, T, V)
        Returns:
            refined: (B, T, V, J, d)
        """
        B, T, V, J, d = tokens.shape
        n_views = V

        # Content + ray Q/K; content values.
        Q = self.q_proj(tokens) + self.q_ray_proj(ray_emb)
        K = self.k_proj(tokens) + self.k_ray_proj(ray_emb)
        values = self.v_proj(tokens)

        # Reshape for multi-head: (B*T*J, n_heads, n_views, head_dim)
        def _reshape(x):
            return x.permute(0, 1, 3, 2, 4).reshape(B * T * J, n_views, self.n_heads, self.head_dim).transpose(1, 2)

        Q = _reshape(Q)  # (BTH, h, V, d_h)
        K = _reshape(K)
        values = _reshape(values)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (BTH, h, V, V)

        if ray_bias is not None:
            # ray_bias: (B, T, V, V, J); permute to (B*T*J, V, V)
            bias = ray_bias.permute(0, 1, 4, 2, 3).reshape(B * T * J, n_views, n_views)
            bias = bias[:, None, :, :].expand(-1, self.n_heads, -1, -1)
            scores = scores + bias

        # Mask
        if view_mask is not None:
            mask = view_mask.bool()  # (B, T, V)
            mask = mask.view(B * T, 1, n_views).expand(-1, n_views, -1)  # (B*T, V, V)
            mask = mask.reshape(B * T, 1, n_views, n_views).expand(-1, J, -1, -1).reshape(B * T * J, 1, n_views, n_views)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, values)  # (BTH, h, V, d_h)
        # Reshape back to (B, T, V, J, d)
        out = out.transpose(1, 2).reshape(B, T, J, n_views, self.n_heads, self.head_dim)
        out = out.reshape(B, T, J, n_views, d)
        out = out.permute(0, 1, 3, 2, 4)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out)

        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * out


class RayConditionedCrossViewAttentionV33(nn.Module):
    """Stack of ray-conditioned cross-view attention layers.

    Parameters
    ----------
    d:
        Model dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of layers.
    dropout:
        Dropout probability.
    use_ray_bias:
        Whether to add an ray-intersection/epipolar bias to attention scores.
    residual_gate_init:
        Initial value of the residual gate (logit).  Default -6 gives near-zero gate.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_ray_bias: bool = True,
        residual_gate_init: float = -6.0,
    ):
        super().__init__()
        self.d = d
        self.ray_emb = _RayEmbedding(d, dropout=dropout)
        self.layers = nn.ModuleList(
            [_RayConditionedCrossViewAttentionLayer(
                d=d,
                n_heads=n_heads,
                dropout=dropout,
                use_ray_bias=False,
                residual_gate_init=residual_gate_init,
            ) for _ in range(n_layers)
            ]
        )
        self.use_ray_bias = use_ray_bias
        if self.use_ray_bias:
            self.sigma_d = nn.Parameter(torch.tensor(0.5))
            self.sigma_a = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        tokens: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d)
            points_2d: (B, T, V, J, 2)
            K: (B, T, V, 3, 3)
            R: (B, T, V, 3, 3)
            t: (B, T, V, 3)
            view_mask: optional (B, T, V)
        Returns:
            refined: (B, T, V, J, d)
        """
        centre, direction = compute_rays(points_2d, K, R, t)  # (B,T,V,3), (B,T,V,J,3)
        ray_emb = self.ray_emb(centre, direction)  # (B, T, V, J, d)

        ray_bias = None
        if self.use_ray_bias:
            # Ray-intersection logit: (B, T, V, V, J)
            ray_bias = ray_intersection_logit(centre, direction, self.sigma_d, self.sigma_a)

        for layer in self.layers:
            tokens = layer(tokens, ray_emb, ray_bias=ray_bias, view_mask=view_mask)
        return tokens
