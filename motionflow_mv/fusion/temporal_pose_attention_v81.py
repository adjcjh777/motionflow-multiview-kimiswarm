"""v81: Lightweight per-joint temporal attention over triangulated 3-D poses.

Adds a single temporal self-attention layer on top of the v25 multi-view
geometry-fusion output. For each joint, the module attends to the same joint
across the temporal clip, producing a smooth, context-aware refinement of the
per-frame pose.

The module is identity-at-init: the output projection is zero-initialised and
the residual gate starts near zero, so it has no effect on the initial v25
estimate.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPoseAttentionV81(nn.Module):
    """Single-layer temporal self-attention over per-joint 3-D positions.

    Parameters
    ----------
    n_joints:
        Number of joints (only used to ensure shape consistency).
    temporal_window:
        Fixed local window size for attention. ``None`` or ``-1`` means the
        whole clip is used.
    residual_gate_init:
        Initial logit of the learnable residual gate. ``-6.0`` gives a near-zero
        contribution at init, warming the new path up gradually.
    dropout:
        Dropout on the output projection.
    """

    def __init__(
        self,
        n_joints: int = 17,
        temporal_window: Optional[int] = 9,
        residual_gate_init: float = -6.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.temporal_window = temporal_window
        self.use_window = temporal_window is not None and temporal_window > 0

        # Single-head temporal attention on 3-D coordinate tokens.
        # We attend over time for each joint independently; the value projection
        # maps the 3-D coordinate to a small embedding and back.
        d_in = 3
        d_hid = max(8, n_joints // 2)
        self.qkv = nn.Linear(d_in, d_hid * 3, bias=False)
        self.out_proj = nn.Linear(d_hid, d_in, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Identity at init: zero the output so the residual path vanishes.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Learnable residual gate so the v81 path is warm-started near zero.
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

    def _window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Return (T, T) boolean mask for the allowed temporal neighbourhood."""
        if not self.use_window or self.temporal_window is None:
            return torch.ones(T, T, dtype=torch.bool, device=device)
        half = self.temporal_window // 2
        t_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        offset = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        return (t_idx - offset).abs() <= half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            x: (B, T, J, 3) triangulated 3-D poses.

        Returns:
            refined: (B, T, J, 3), same shape as input.
        """
        B, T, J, _ = x.shape

        # Scale-independent attention: use the 3-D coordinate itself as the
        # token. Each joint is treated independently.
        # (B, T, J, 3) -> separate per-joint tokens.
        qkv = self.qkv(x)  # (B, T, J, 3*d_hid)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, J, d_hid)

        # Compute attention over the temporal dimension for each joint.
        # q, k: (B, J, T, d_hid)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scores: (B, J, T, T)
        d_hid = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_hid)

        # Mask out-of-window positions.
        mask = self._window_mask(T, x.device)  # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Aggregate and project back to 3-D.
        out = torch.matmul(attn, v)  # (B, J, T, d_hid)
        out = out.permute(0, 2, 1, 3)  # (B, T, J, d_hid)
        out = self.out_proj(out)  # (B, T, J, 3)
        out = self.dropout(out)

        # Gated residual update; start near identity so v81 is safe at init.
        gate = torch.sigmoid(self.residual_gate)
        return x + gate * out
