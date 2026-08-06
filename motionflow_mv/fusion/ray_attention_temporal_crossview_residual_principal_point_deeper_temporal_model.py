"""Hierarchical view -> deeper temporal -> skeleton-joint attention fusion.

Builds on the hierarchical attention model by replacing the shallow temporal
attention stage with a deeper, residual-gated temporal transformer.  Each
temporal sub-layer keeps a learnable residual gate (initialised near zero) and a
per-layer temporal positional embedding.  This lets us safely stack more
temporal layers without the training instability that usually comes with deeper
attention blocks.
"""

import torch
import torch.nn as nn

from .ray_attention_hierarchical_view_temporal_joint_residual_principal_point_model import (
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint,
    _HierarchicalViewTemporalJointBlock,
)


class _DeeperTemporalBlock(nn.Module):
    """Deeper residual-gated temporal attention block.

    Parameters
    ----------
    d:
        Token dimension.
    n_layers:
        Number of stacked temporal transformer encoder layers.
    n_heads:
        Number of attention heads.
    max_temporal_len:
        Maximum clip length for the per-layer positional embeddings.
    gate_init:
        Initial value for the learnable residual gates.
    """

    def __init__(
        self,
        d: int,
        n_layers: int,
        n_heads: int,
        max_temporal_len: int = 256,
        gate_init: float = 1e-3,
    ):
        super().__init__()
        self.d = d
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.time_pos_embed = nn.ParameterList(
            [
                nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
                for _ in range(n_layers)
            ]
        )
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(gate_init)) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated deeper temporal attention.

        Args
        ----
        x:
            ``(N, T, d)`` tokens, where ``N = B*V*J``.

        Returns
        -------
        ``(N, T, d)`` refined tokens.
        """
        T = x.shape[1]
        for layer, pos, gate in zip(self.layers, self.time_pos_embed, self.gates):
            residual = x
            x = x + pos[:T].unsqueeze(0)
            out = layer(x)
            x = residual + gate * out
        return x


class _HierarchicalViewDeeperTemporalJointBlock(_HierarchicalViewTemporalJointBlock):
    """Hierarchical block with a deeper residual-gated temporal stage."""

    def __init__(
        self,
        d: int,
        n_views: int,
        n_view_groups: int,
        n_heads: int,
        n_view_layers: int,
        n_temporal_layers: int,
        n_joint_graph_layers: int,
        max_temporal_len: int = 256,
    ):
        super().__init__(
            d=d,
            n_views=n_views,
            n_view_groups=n_view_groups,
            n_heads=n_heads,
            n_view_layers=n_view_layers,
            n_temporal_layers=0,
            n_joint_graph_layers=n_joint_graph_layers,
        )
        self.temporal_layers = nn.ModuleList(
            [
                _DeeperTemporalBlock(
                    d=d,
                    n_layers=n_temporal_layers,
                    n_heads=n_heads,
                    max_temporal_len=max_temporal_len,
                )
            ]
        )


class RayAttentionFusionModelHierarchicalViewDeeperTemporalResidualPrincipalPoint(
    RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint
):
    """Hierarchical attention with a deeper, residual-gated temporal stage.

    This is a drop-in replacement for
    ``RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint``
    that uses a deeper temporal transformer block.  The extra capacity is aimed
    at capturing longer-range motion dynamics, which should lower MPJPE on
    fast/articulated motion.

    Parameters
    ----------
    n_temporal_layers:
        Number of layers inside the deeper temporal block (default 4).
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        return_raw: bool = False,
        n_view_groups: int = 2,
        n_view_layers: int = 2,
        n_temporal_layers: int = 4,
        n_joint_graph_layers: int = 1,
        use_skeleton_graph: bool = True,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=return_visibility,
            return_raw=return_raw,
            n_view_groups=n_view_groups,
            n_view_layers=n_view_layers,
            n_temporal_layers=n_temporal_layers,
            n_joint_graph_layers=n_joint_graph_layers,
            use_skeleton_graph=use_skeleton_graph,
        )
        self.hierarchical_block = _HierarchicalViewDeeperTemporalJointBlock(
            d=d,
            n_views=n_views,
            n_view_groups=n_view_groups,
            n_heads=n_heads,
            n_view_layers=n_view_layers,
            n_temporal_layers=n_temporal_layers,
            n_joint_graph_layers=n_joint_graph_layers,
            max_temporal_len=max_temporal_len,
        )
