"""v82: Multi-scale per-joint temporal attention over triangulated 3-D poses.

Builds on v81 by running several temporal-attention branches with different
(temporal) receptive fields, then adaptively fusing them with a lightweight
per-joint scale selector.  The extra scales let the model capture both fast
local dynamics (small window) and slow global motion (full clip) without an
increase in the number of parameters proportional to clip length.

The module is identity-at-init: every output projection is zero-initialised,
and the residual gate starts near zero, so it has no effect on the initial
v25 estimate.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPoseAttentionV82(nn.Module):
    """Multi-scale temporal self-attention over per-joint 3-D positions.

    Parameters
    ----------
    n_joints:
        Number of joints (only used to ensure shape consistency).
    temporal_windows:
        List of window sizes, one per branch. ``None`` or ``-1`` or ``<=0`` in a
        branch means the whole clip is used for that branch.
    hidden_dim:
        Hidden dimension of the joint-coordinate QKV projection.
    dropout:
        Dropout on each branch output projection.
    residual_gate_init:
        Initial logit of the learnable residual gate. ``-6.0`` gives a near-zero
        contribution at init, warming the new path up gradually.
    """

    def __init__(
        self,
        n_joints: int = 17,
        temporal_windows: Sequence[Optional[int]] = (5, 13, -1),
        hidden_dim: int = 16,
        dropout: float = 0.1,
        residual_gate_init: float = -6.0,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_scales = len(temporal_windows)
        self.temporal_windows = list(temporal_windows)
        self.hidden_dim = hidden_dim

        d_in = 3
        # Shared QKV keeps parameter count tiny; each branch has its own value
        # projection head to specialise on its temporal scale.
        self.qkv = nn.Linear(d_in, hidden_dim * 3, bias=False)

        # Per-scale output projections are zero-initialised for identity init.
        self.out_projs = nn.ModuleList(
            nn.Linear(hidden_dim, d_in, bias=True) for _ in range(self.n_scales)
        )
        for out_proj in self.out_projs:
            nn.init.zeros_(out_proj.weight)
            if out_proj.bias is not None:
                nn.init.zeros_(out_proj.bias)

        self.dropout = nn.Dropout(dropout)

        # Per-joint scale selector: small MLP operating on the per-joint
        # coordinate norm to produce a weight for each scale.  Starts near zero
        # so the residual contribution vanishes at init.
        self.scale_selector = nn.Sequential(
            nn.Linear(d_in, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_scales, bias=True),
        )
        # Initialise selector to near-zero output -> all scales suppressed.
        nn.init.zeros_(self.scale_selector[0].weight)
        nn.init.zeros_(self.scale_selector[0].bias)
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)

        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

    def _window_mask(self, T: int, window: Optional[int], device: torch.device) -> torch.Tensor:
        """Return (T, T) boolean mask for a temporal window."""
        if window is None or window <= 0:
            return torch.ones(T, T, dtype=torch.bool, device=device)
        half = window // 2
        t_idx = torch.arange(T, device=device).unsqueeze(1)
        offset = torch.arange(T, device=device).unsqueeze(0)
        return (t_idx - offset).abs() <= half

    def _attend(
        self,
        x: torch.Tensor,
        window: Optional[int],
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """Compute a single temporal-attention branch.

        Args:
            x: (B, T, J, 3)
            window: window size for this branch
            out_proj: output projection for this branch

        Returns:
            (B, T, J, 3)
        """
        B, T, J, _ = x.shape
        qkv = self.qkv(x)  # (B, T, J, 3*hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, J, hidden_dim)

        # Per joint, over time.
        q = q.permute(0, 2, 1, 3)  # (B, J, T, hidden_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        mask = self._window_mask(T, window, x.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # (B, J, T, hidden_dim)
        out = out.permute(0, 2, 1, 3)  # (B, T, J, hidden_dim)
        out = out_proj(out)  # (B, T, J, 3)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale temporal attention.

        Args:
            x: (B, T, J, 3) triangulated 3-D poses.

        Returns:
            refined: (B, T, J, 3), same shape as input.
        """
        # Per-scale branch outputs.
        branch_outs = []
        for window, out_proj in zip(self.temporal_windows, self.out_projs):
            branch_outs.append(self._attend(x, window, out_proj))
        # stack -> (n_scales, B, T, J, 3)
        stacked = torch.stack(branch_outs, dim=0)

        # Per-joint scale selector based on the raw joint position.
        # scale_weights: (B, T, J, n_scales)
        scale_weights = self.scale_selector(x)
        scale_weights = F.softmax(scale_weights, dim=-1)

        # Weighted fusion of scales.
        # Reshape for broadcasting: (1, B, T, J, n_scales) -> (n_scales, B, T, J, 1)
        weights = scale_weights.permute(3, 0, 1, 2).unsqueeze(-1)
        fused = (weights * stacked).sum(dim=0)  # (B, T, J, 3)
        fused = self.dropout(fused)

        gate = torch.sigmoid(self.residual_gate)
        return x + gate * fused
