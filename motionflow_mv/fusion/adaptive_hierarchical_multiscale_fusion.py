"""Adaptive scale-selective hierarchical multi-scale fusion.

This module is a drop-in replacement for
``_HierarchicalMultiscaleFusion`` used in ``OmniMultiViewFusionV3/4/5``.
For each temporal/joint scale a cross-view transformer is applied, and the
resulting scale-specific features are fused with *per-token* scale attention
instead of a fixed linear fusion.  The query for the scale attention is the
original full-resolution token, so the model can adaptively down-weight scales
that are uninformative for each individual (time, view, joint) token.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHierarchicalMultiscaleFusion(nn.Module):
    """Adaptive scale-selective hierarchical multi-scale fusion.

    Args:
        d: token dimension.
        n_views: number of camera views.
        scales: temporal / joint downsample factors.
        n_heads: attention heads for the cross-view transformer layers.
        dropout: dropout applied inside the transformer layers.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(scales)
        if any(s < 1 for s in self.scales):
            raise ValueError("All scale factors must be >= 1")

        # One cross-view transformer per scale (same as the fixed-fusion version).
        self.branches = nn.ModuleList()
        for _ in self.scales:
            self.branches.append(
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
            )

        # Per-token scale attention.
        self.scale_query = nn.Linear(d, d)
        self.scale_keys = nn.ModuleList([nn.Linear(d, d) for _ in self.scales])
        self.scale_attention_temperature = d ** -0.5

        # Optional residual projection after weighted sum.
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, V, J, d). Returns: (B, T, V, J, d)."""
        B, T, V, J, d = x.shape
        x_in = x

        scale_features = []
        for scale, layer in zip(self.scales, self.branches):
            if scale == 1:
                x_s = x
            else:
                t_target = max(1, T // scale)
                x_s = x.permute(0, 2, 3, 4, 1).reshape(B * V * J, d, T)
                x_s = F.adaptive_avg_pool1d(x_s, t_target)
                x_s = x_s.view(B, V, J, d, t_target).permute(0, 4, 1, 2, 3)

            t_cur = x_s.shape[1]

            if scale > 1:
                j_target = max(1, J // scale)
                x_s = x_s.permute(0, 1, 2, 4, 3).reshape(B * t_cur * V, d, J)
                x_s = F.adaptive_avg_pool1d(x_s, j_target)
                x_s = x_s.view(B, t_cur, V, d, j_target).permute(0, 1, 2, 4, 3)

            j_cur = x_s.shape[3]

            # Cross-view attention.
            x_s = x_s.permute(0, 1, 3, 2, 4).reshape(B * t_cur * j_cur, V, d)
            x_s = layer(x_s)
            x_s = x_s.view(B, t_cur, j_cur, V, d).permute(0, 1, 3, 2, 4)

            # Upsample joints back to J.
            if scale > 1:
                x_s = x_s.permute(0, 1, 3, 4, 2).reshape(B * t_cur * V, d, j_cur)
                x_s = F.interpolate(x_s, size=J, mode="linear", align_corners=False)
                x_s = x_s.view(B, t_cur, V, d, J).permute(0, 1, 2, 4, 3)

            # Upsample time back to T.
            if scale > 1:
                x_s = x_s.permute(0, 2, 3, 4, 1).reshape(B * V * J, d, t_cur)
                x_s = F.interpolate(x_s, size=T, mode="linear", align_corners=False)
                x_s = x_s.view(B, V, J, d, T).permute(0, 4, 1, 2, 3)

            scale_features.append(x_s)

        # Per-token scale attention.
        # q: (B, T, V, J, d)
        q = self.scale_query(x_in)  # (B, T, V, J, d)
        # keys: list of (B, T, V, J, d) -> stack to (B, T, V, J, S, d)
        keys = torch.stack([proj(f) for proj, f in zip(self.scale_keys, scale_features)], dim=4)
        # q_uns = (B, T, V, J, 1, d), keys = (B, T, V, J, S, d)
        q_uns = q.unsqueeze(4)
        attn = (q_uns * keys).sum(dim=-1) * self.scale_attention_temperature  # (B, T, V, J, S)
        attn = F.softmax(attn, dim=-1)
        # weighted sum
        feats = torch.stack(scale_features, dim=4)  # (B, T, V, J, S, d)
        attn = attn.unsqueeze(-1)  # (B, T, V, J, S, 1)
        x_out = (attn * feats).sum(dim=4)  # (B, T, V, J, d)
        x_out = self.out_proj(x_out)
        return self.norm(x_out + x_in)
