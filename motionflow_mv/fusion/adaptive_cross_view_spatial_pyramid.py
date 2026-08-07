"""Adaptive multi-scale cross-view spatial pyramid.

The module extends the static ``CrossViewSpatialPyramid`` in
``cross_view_spatial_pyramid.py`` by learning input-dependent scale weights.
Instead of concatenating all fixed-resolution branches, a lightweight gating
network first pools global statistics from the per-view/joint features and
predicts a soft distribution over the requested scales.  The scale branches are
 then fused as a weighted combination with a residual connection.

This lets the model adapt its receptive field on the joint axis per sample:
- Fine scales preserve local joint detail.
- Coarser scales capture limb/torso-level context.
- The gate can selectively emphasize the useful scales for each input.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCrossViewSpatialPyramid(nn.Module):
    """Input-adaptive cross-view spatial pyramid.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views.
    scales:
        Downsample factors for the joint axis.  Default ``(1, 2, 4)``.
    n_heads:
        Number of attention heads in each cross-view block.
    gate_hidden:
        Hidden dimension of the scale-gating MLP.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 1,
        gate_hidden: int | None = None,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(scales)
        if any(s < 1 for s in self.scales):
            raise ValueError("All scale factors must be >= 1")

        from .cross_view_spatial_pyramid import _CrossViewBlock

        self.branches = nn.ModuleList(
            [_CrossViewBlock(d, n_views, n_heads=n_heads) for _ in self.scales]
        )

        gate_hidden = gate_hidden or max(16, d // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(self.scales)),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def _gate_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample adaptive scale weights.

        Args
        ----
        x:
            Tensor of shape ``(N, V, J, d)``.

        Returns
        -------
        weights:
            Softmax weights of shape ``(N, S)``.
        """
        # Pool global mean and std over views/joints for each sample.
        mean_feat = x.mean(dim=(1, 2))  # (N, d)
        std_feat = x.std(dim=(1, 2), unbiased=False)  # (N, d)
        stats = torch.cat([mean_feat, std_feat], dim=-1)  # (N, d*2)
        logits = self.gate_mlp(stats)  # (N, S)
        return F.softmax(logits, dim=-1)

    def _run_branch(self, x: torch.Tensor, scale: int, branch: nn.Module) -> torch.Tensor:
        """Run a single scale branch and return upsampled features.

        Args
        ----
        x:
            Input tensor of shape ``(N, V, J, d)``.
        scale:
            Downsample factor for this branch.
        branch:
            The cross-view block for this scale.

        Returns
        -------
        Tensor of shape ``(N, V, J, d)``.
        """
        N, V, J, d = x.shape
        if scale == 1:
            x_s = x.permute(0, 2, 1, 3).reshape(N * J, V, d)
            x_s = branch(x_s)
            x_s = x_s.view(N, J, V, d).permute(0, 2, 1, 3)
        else:
            target_j = max(1, J // scale)
            x_perm = x.permute(0, 1, 3, 2).reshape(N * V, d, J)
            x_pooled = F.adaptive_avg_pool1d(x_perm, target_j)
            x_pooled = x_pooled.view(N, V, d, target_j).permute(0, 3, 1, 2)
            x_pooled = x_pooled.reshape(N * target_j, V, d)
            x_attended = branch(x_pooled)
            x_attended = x_attended.view(N, target_j, V, d).permute(0, 2, 3, 1)
            x_attended = x_attended.reshape(N * V, d, target_j)
            x_upsampled = F.interpolate(x_attended, size=J, mode="linear", align_corners=False)
            x_s = x_upsampled.view(N, V, d, J).permute(0, 1, 3, 2)
        return x_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        x:
            Tensor of shape ``(N, V, J, d)``.

        Returns
        -------
        Tensor of shape ``(N, V, J, d)``.
        """
        x_in = x
        scale_features = []
        for scale, branch in zip(self.scales, self.branches):
            scale_features.append(self._run_branch(x, scale, branch))

        # Adaptive soft scale selection per sample.
        gate = self._gate_vector(x)  # (N, S)
        # Stack to (S, N, V, J, d), weight by gate, and sum.
        stacked = torch.stack(scale_features, dim=0)  # (S, N, V, J, d)
        gate = gate.view(-1, *([1] * (stacked.ndim - 2)), len(self.scales))
        gate = gate.permute(4, 0, 1, 2, 3)  # (S, N, 1, 1, 1)
        x_fused = (gate * stacked).sum(dim=0)  # (N, V, J, d)

        x_out = self.fusion(x_fused)
        x_out = self.norm(x_out + x_in)
        return x_out
