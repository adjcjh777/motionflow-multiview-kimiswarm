"""Temporal refinement of 3D skeletons over a sliding window.

Given a window of DLT triangulated poses and per-view 2D observations,
use a small GRU to refine the center frame. The model can suppress
single-frame outliers by borrowing information from neighboring frames.
"""

import torch
import torch.nn as nn


class TemporalRefinerModel(nn.Module):
    """GRU-based temporal refiner.

    Input:
        x:           (B, T, V, J, 3)  -- per-view (x, y, confidence)
        baseline_3d: (B, T, J, 3)     -- DLT poses for each frame in window
    Output:
        (B, J, 3) refined 3D pose for the center frame
    """

    def __init__(self, j: int = 17, d: int = 64, n_views: int = 5, hidden: int = 128):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.hidden = hidden
        self.lift = nn.Linear(3, d)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
        self.frame_feat_size = d + 3
        self.gru = nn.GRU(self.frame_feat_size, hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def _per_frame_features(self, x: torch.Tensor, baseline: torch.Tensor):
        """x: (B, V, J, 3); baseline: (B, J, 3). Returns (B, J, d+3)."""
        B, V, J, _ = x.shape
        emb = self.lift(x)  # (B, V, J, d)
        emb = emb.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
        attn_out, _ = self.attn(emb, emb, emb)  # (B*J, V, d)
        feat = attn_out.mean(dim=1).view(B, J, self.d)
        return torch.cat([feat, baseline], dim=-1)  # (B, J, d+3)

    def forward(self, x: torch.Tensor, baseline_3d: torch.Tensor) -> torch.Tensor:
        """x: (B, T, V, J, 3); baseline_3d: (B, T, J, 3)."""
        B, T, V, J, _ = x.shape
        # Build per-frame per-joint features.
        feats = []
        for t in range(T):
            feat = self._per_frame_features(x[:, t], baseline_3d[:, t])  # (B, J, d+3)
            feats.append(feat)
        feats = torch.stack(feats, dim=1)  # (B, T, J, d+3)
        # Process each joint independently with shared GRU.
        feats = feats.permute(0, 2, 1, 3).reshape(B * J, T, self.frame_feat_size)
        out, _ = self.gru(feats)  # (B*J, T, 2*hidden)
        center = T // 2
        residual = self.mlp(out[:, center])  # (B*J, 3)
        residual = residual.view(B, J, 3)
        return baseline_3d[:, center] + residual
