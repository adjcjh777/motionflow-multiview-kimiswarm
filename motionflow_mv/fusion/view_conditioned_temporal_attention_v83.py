"""v83: View-conditioned temporal attention over per-view ray tokens.

Operates *before* triangulation on the ray tokens produced by
``MultiViewGeometryFusionV25``.  For each view and joint, it first runs a
masked temporal self-attention over the token sequence, then performs a
cross-view attention step biased by per-view reliability.  The whole block
is gated by a learnable residual gate initialised near zero, so it is an
identity no-op at the start of training and preserves the v25 baseline.

Parameters match the v83 design proposal in ``docs/v83_design_20260812.md``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalAttention(nn.Module):
    """Per-view/per-joint temporal self-attention with a local window mask."""

    def __init__(self, d: int, n_heads: int, dropout: float, temporal_window: Optional[int]):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.temporal_window = temporal_window
        self.use_window = temporal_window is not None and temporal_window > 0

        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Identity-at-init: zero output projection so the residual vanishes.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Return (T, T) boolean mask for the allowed temporal neighbourhood."""
        if not self.use_window:
            return torch.ones(T, T, dtype=torch.bool, device=device)
        half = self.temporal_window // 2
        t_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        offset = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        return (t_idx - offset).abs() <= half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, V, J, d)

        Returns:
            out: (B, T, V, J, d)
        """
        B, T, V, J, d = x.shape
        # Treat (B, V, J) as batch, T as sequence: (B*V*J, T, d).
        x = x.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, d)

        qkv = self.qkv(x)  # (B*V*J, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*V*J, T, d)

        # Multi-head reshape: (N, h, T, d_h)
        q = q.view(B * V * J, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B * V * J, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B * V * J, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self._window_mask(T, x.device)  # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (N, h, T, d_h)
        out = out.transpose(1, 2).reshape(B * V * J, T, d)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Reshape back to (B, T, V, J, d)
        out = out.reshape(B, V, J, T, d).permute(0, 3, 1, 2, 4)
        return out


class _CrossViewAttention(nn.Module):
    """Cross-view attention with optional per-view reliability bias."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        view_reliability: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        use_view_reliability_bias: bool = True,
    ) -> torch.Tensor:
        """Args:
            x: (B, T, V, J, d)
            view_reliability: (B, T, V, J)
            view_mask: optional (B, T, V)

        Returns:
            out: (B, T, V, J, d)
        """
        B, T, V, J, d = x.shape
        # Treat (B, T, J) as batch, V as sequence: (B*T*J, V, d).
        x = x.reshape(B * T * J, V, d)

        qkv = self.qkv(x)  # (B*T*J, V, 3d)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*T*J, V, d)

        q = q.view(B * T * J, V, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B * T * J, V, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B * T * J, V, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: (B*T*J, h, V, V)

        if use_view_reliability_bias:
            # reliability bias depends on the key view.
            # view_reliability: (B, T, V, J) -> (B*T*J, V) in the same flatten order.
            rel = view_reliability.permute(0, 1, 3, 2).reshape(B * T * J, V)  # (B*T*J, V)
            log_rel = torch.log(rel.clamp(min=1e-6))
            scores = scores + log_rel[:, None, None, :]  # broadcast over h and query view

        if view_mask is not None:
            # view_mask: (B, T, V) -> (B*T*J, V) mask for keys.
            vm = view_mask.permute(0, 2, 1).reshape(B * T, V)
            vm = vm.unsqueeze(1).expand(-1, J, -1).reshape(B * T * J, V)
            scores = scores.masked_fill(~vm[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B*T*J, h, V, d_h)
        out = out.transpose(1, 2).reshape(B * T * J, V, d)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Reshape back to (B, T, V, J, d)
        out = out.reshape(B, T, J, V, d).permute(0, 1, 3, 2, 4)
        return out


class ViewConditionedTemporalAttentionV83(nn.Module):
    """View-conditioned temporal attention over per-view ray tokens.

    Parameters
    ----------
    d:
        Token dimension (must match the v25 ray-token dimension).
    n_heads:
        Number of attention heads for both temporal and cross-view attention.
    n_views:
        Number of views (used for shape hints; currently unused).
    temporal_window:
        Local temporal window size. ``None`` or ``<=0`` means full-clip.
    n_layers:
        Number of stacked [temporal, cross-view] attention blocks.
    dropout:
        Dropout probability on the output projections.
    residual_gate_init:
        Initial logit of the global residual gate. ``-6.0`` gives a near-zero
        contribution at init, preserving the v25 baseline.
    use_view_reliability_bias:
        If True, add a log-reliability bias to the cross-view attention scores.
        If the supplied reliability is uniform, the bias vanishes and the block
        reduces to plain per-view temporal attention plus cross-view attention.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        temporal_window: int = 9,
        n_layers: int = 1,
        dropout: float = 0.1,
        residual_gate_init: float = -6.0,
        use_view_reliability_bias: bool = True,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.n_layers = n_layers
        self.use_view_reliability_bias = use_view_reliability_bias

        # Stack n_layers of [temporal, cross-view] blocks. Each layer owns its own
        # parameters so that n_layers > 1 actually does extra work.
        self.layers = nn.ModuleList()
        for _ in range(max(n_layers, 1)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "temporal": _TemporalAttention(d, n_heads, dropout, temporal_window),
                        "cross_view": _CrossViewAttention(d, n_heads, dropout),
                    }
                )
            )

        # Global residual gate; sigmoid(-6.0) ~ 0.002, so the module is near-identity.
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float32))

    def _normalize_reliability(
        self,
        view_reliability: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-view-joint reliability (B, T, V, J)."""
        B, T, V, J, _ = tokens.shape
        if view_reliability.dim() == 3:
            # (B, T, V) -> (B, T, V, J)
            return view_reliability.unsqueeze(-1).expand(-1, -1, -1, J)
        if view_reliability.dim() == 4:
            # (B, T, V, J) already.
            return view_reliability
        raise ValueError(
            f"view_reliability must be (B, T, V) or (B, T, V, J); got {view_reliability.shape}"
        )

    def forward(
        self,
        tokens: torch.Tensor,
        view_reliability: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply view-conditioned temporal attention.

        Args:
            tokens: (B, T, V, J, d)
            view_reliability: (B, T, V) or (B, T, V, J)
            view_mask: optional (B, T, V)

        Returns:
            refined: (B, T, V, J, d)
        """
        rel = self._normalize_reliability(view_reliability, tokens)
        gate = torch.sigmoid(self.residual_gate)

        out = tokens
        for layer in self.layers:
            # Temporal attention over time for each view/joint.
            temporal_out = layer["temporal"](out)
            out = out + gate * temporal_out

            # Cross-view reliability-biased attention for each time/joint.
            cross_out = layer["cross_view"](
                out,
                rel,
                view_mask=view_mask,
                use_view_reliability_bias=self.use_view_reliability_bias,
            )
            out = out + gate * cross_out

        return out
