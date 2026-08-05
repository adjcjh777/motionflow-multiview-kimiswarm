"""Attention fusion model with camera parameter input.

Extends the minimal model by feeding each view's projection matrix
as an additional cue, helping the network learn geometry-aware fusion.
"""

import torch
import torch.nn as nn
from .attention import ViewAttentionFusion


class AttentionFusionModelV2(nn.Module):
    """Trainable fusion of per-view 2D keypoints + confidence + camera into 3D pose.

    Input:
        x:       (B, V, J, 3)  -> (x, y, confidence)
        cameras: (B, V, 12)     -> flattened projection matrices [P11, ..., P34]
    Output:
        (B, J, 3) 3D joint positions in world space.
    """

    def __init__(self, j: int = 17, d: int = 32, n_views: int = 4):
        super().__init__()
        self.j = j
        self.d = d
        self.lift = nn.Linear(3, d)
        self.cam_embed = nn.Linear(12, d)
        self.attention = ViewAttentionFusion(d=d, j=j)
        self.head = nn.Linear(d, 3)

    def forward(self, x: torch.Tensor, cameras: torch.Tensor) -> torch.Tensor:
        # x: (B, V, J, 3), cameras: (B, V, 12)
        B, V, J, _ = x.shape
        x = self.lift(x)  # (B, V, J, D)
        cam = self.cam_embed(cameras)  # (B, V, D)
        cam = cam.unsqueeze(2).expand(-1, -1, J, -1)  # (B, V, J, D)
        x = x + cam
        x = self.attention(x)  # (B, J, D)
        return self.head(x)  # (B, J, 3)
