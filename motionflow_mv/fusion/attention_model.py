"""Minimal trainable attention fusion model: multi-view 2D -> 3D pose."""

import torch
import torch.nn as nn
from .attention import ViewAttentionFusion


class AttentionFusionModel(nn.Module):
    """Trainable fusion of per-view 2D keypoints + confidence into 3D skeleton.

    Input:  (B, V, J, 3) where last dim is (x, y, confidence)
    Output: (B, J, 3) 3D joint positions in world space.
    """

    def __init__(self, j: int = 17, d: int = 32, n_views: int = 4):
        super().__init__()
        self.j = j
        self.d = d
        self.lift = nn.Linear(3, d)
        self.attention = ViewAttentionFusion(d=d, j=j)
        self.head = nn.Linear(d, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, J, 3)
        B, V, J, _ = x.shape
        x = self.lift(x)  # (B, V, J, D)
        x = self.attention(x)  # (B, J, D)
        return self.head(x)  # (B, J, 3)
