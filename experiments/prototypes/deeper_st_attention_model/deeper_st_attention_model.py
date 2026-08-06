"""Deeper spatio-temporal (ST) attention prototype.

Builds on :class:`SpatiotemporalPrincipalPointModel` by replacing its
single-pass factorised (temporal x view x joint) attention with a stack of
interleaved residual blocks.  Each block applies one
:class:`nn.TransformerEncoderLayer` along the temporal, view, and joint axes
in sequence, and the whole block is wrapped in a residual connection.

This is a throw-away prototype for the next-iteration swarm; it keeps the same
input/output contract as the existing principal-point models so that it can
slot into the current training/eval harness once GPU time is available.
"""

from typing import Optional

import torch
import torch.nn as nn

from motionflow_mv.models.spatiotemporal_principal_point_model import (
    SpatiotemporalPrincipalPointModel,
)


class _StAttentionBlock(nn.Module):
    """One interleaved T-V-J TransformerEncoderLayer block."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.view = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.joint = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply T, V, J attention in sequence.

        Input / output shape: (B, T, V, J, d)
        """
        B, T, V, J, d = feat.shape

        # Temporal axis: (B, V, J, T, d) -> (B*V*J, T, d)
        x = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, d)
        x = self.temporal(x)
        x = x.view(B, V, J, T, d).permute(0, 3, 1, 2, 4)

        # View axis: (B, T, J, V, d) -> (B*T*J, V, d)
        x = x.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        x = self.view(x)
        x = x.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)

        # Joint axis: (B, T, V, J, d) -> (B*T*V, J, d)
        x = x.reshape(B * T * V, J, d)
        x = self.joint(x)
        x = x.view(B, T, V, J, d)

        return x


class DeeperStAttentionPrincipalPointModel(SpatiotemporalPrincipalPointModel):
    """Deeper factorised ST attention model with residual blocks.

    Parameters
    ----------
    n_st_blocks:
        Number of interleaved T-V-J attention blocks (default 4).
    See ``SpatiotemporalPrincipalPointModel`` for the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_st_blocks: int = 4,
        max_temporal_len: int = 256,
        residual_hidden: Optional[int] = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
    ):
        # Disable the base model's single-pass factorised layers; we replace
        # them with the deeper residual blocks defined below.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_temporal_layers=0,
            n_view_layers=0,
            n_joint_layers=0,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
        )
        self.n_st_blocks = n_st_blocks
        self.st_blocks = nn.ModuleList(
            [_StAttentionBlock(d, n_heads) for _ in range(n_st_blocks)]
        )

    def _factorised_attention(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply ``n_st_blocks`` interleaved T-V-J attention blocks.

        Each block is wrapped in a residual connection to stabilise the deeper
        stack.
        """
        for block in self.st_blocks:
            feat = feat + block(feat)
        return feat
