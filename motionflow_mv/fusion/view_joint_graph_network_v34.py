"""v34: View-Joint Graph Network for cross-view fusion.

Wraps the existing ``CrossViewGraphAttention`` prototype and makes it
variable-view safe by zero-masking padded views and rebuilding the
skeleton graph on first forward.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    CrossViewGraphAttention,
)
from motionflow_mv.models.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)


def _skeleton_for_joints(j: int) -> Tuple[list, list]:
    """Return parents and symmetry pairs for a given joint count."""
    if j == 17:
        return list(H36M_17_PARENTS), list(H36M_17_SYMMETRY_PAIRS)
    if j == 28:
        return list(MPI_INF_3DHP_28_PARENTS), list(MPI_INF_3DHP_28_SYMMETRY_PAIRS)
    # Fallback: chain parents, no symmetry.
    return [-1] + list(range(j - 1)), []


class ViewJointGraphNetworkV34(nn.Module):
    """Variable-view view-joint graph attention block.

    Args:
        d: token dimension.
        n_views: maximum number of padded views (e.g. 14).
        n_layers: number of graph attention layers.
        n_heads: attention heads per layer.
        dropout: dropout on attention weights.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 14,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_layers = n_layers

        # The prototype builds the graph lazily; we will rebuild per forward.
        self.graph_attn = CrossViewGraphAttention(
            d=d,
            n_views=n_views,
            n_layers=n_layers,
            n_heads=n_heads,
            n_edge_types=4,
            dropout=dropout,
        )

        # Output projection zeroed at init for identity behaviour.
        self.out_proj = nn.Linear(d, d)
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        # Gated residual: near-zero at init.
        self.residual_gate = nn.Parameter(torch.tensor(-6.0))

        # Cached graph per (joints) signature.
        self._graph_cache: dict = {}

    def _get_graph(self, j: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if j not in self._graph_cache:
            parents, symmetry = _skeleton_for_joints(j)
            edge_index, edge_type = self.graph_attn.build_edge_index(
                j=j,
                parents=parents,
                symmetry_pairs=symmetry,
                add_self_loops=True,
            )
            self._graph_cache[j] = (edge_index.to(device), edge_type.to(device))
        return self._graph_cache[j]

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            view_mask: optional (B, T, V) bool.
        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        edge_index, edge_type = self._get_graph(J, tokens.device)

        # Flatten batch/time for the graph: (B*T, V, J, d).
        x = tokens.reshape(B * T, V, J, d)

        # Zero out padded views so they do not propagate.
        if view_mask is not None:
            x = x * view_mask.reshape(B * T, V, 1, 1).float()

        out = self.graph_attn(x, edge_index=edge_index, edge_type=edge_type)
        out = self.out_proj(out)

        # Re-apply mask after graph update.
        if view_mask is not None:
            out = out * view_mask.reshape(B * T, V, 1, 1).float()

        out = out.view(B, T, V, J, d)
        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * out
