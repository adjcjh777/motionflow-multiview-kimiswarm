"""Residual refinement on top of a baseline 3D skeleton.

Given a coarse triangulated 3D pose (e.g. DLT) and the multi-view 2D
observations, predict a small residual correction that reduces
reprojection error and removes outlier frames.
"""

import torch
import torch.nn as nn


class ResidualRefinerModel(nn.Module):
    """Per-frame residual refinement of a 3D skeleton.

    Input:
        x:            (B, V, J, 3)  -- per-view (x, y, confidence)
        baseline_3d:  (B, J, 3)     -- coarse 3D pose (e.g. DLT)
    Output:
        (B, J, 3) refined 3D pose = baseline_3d + delta
    """

    def __init__(self, j: int = 17, d: int = 64, n_views: int = 5):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.lift = nn.Linear(3, d)
        # Aggregate per-view embeddings with attention across views.
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        # Predict a small residual correction from per-joint features + baseline.
        self.mlp = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

    def forward(self, x: torch.Tensor, baseline_3d: torch.Tensor) -> torch.Tensor:
        B, V, J, _ = x.shape
        # Normalize 2D coordinates to avoid dominating the 3D baseline features.
        x = x.clone()
        x[..., :2] = x[..., :2] / 1000.0
        x_emb = self.lift(x)  # (B, V, J, D)
        x_emb = x_emb.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)  # (B*J, V, D)
        # Mean pool over views and reshape back.
        features = attn_out.mean(dim=1).view(B, J, self.d)  # (B, J, D)
        concat = torch.cat([features, baseline_3d], dim=-1)  # (B, J, D+3)
        delta = self.mlp(concat)
        return baseline_3d + delta
