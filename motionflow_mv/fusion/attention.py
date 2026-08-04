"""Lightweight view-attention fusion for multi-view 3D human pose."""

import math
import torch
import torch.nn as nn


class ViewAttentionFusion(nn.Module):
    """Per-joint attention over views.

    Input:  (B, V, J, D) per-view features (e.g. lifted 2D coordinates + confidence)
    Output: (B, J, D) fused features.

    Reference: docs/swarm_iter1/08_attention_fusion.md
    """

    def __init__(self, d: int, j: int):
        super().__init__()
        self.d = d
        self.j = j
        self.query = nn.Parameter(torch.randn(j, d) * 0.02)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, J, D)
        B, V, J, D = x.shape
        q = self.query.unsqueeze(0)  # (1, J, D)
        k = self.Wk(x)               # (B, V, J, D)
        v = self.Wv(x)               # (B, V, J, D)
        # attention scores per joint over views
        scores = torch.einsum("b v j d, q j d -> b j v", k, q) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)  # (B, J, V)
        out = torch.einsum("b v j d, b j v -> b j d", v, attn)
        return out
