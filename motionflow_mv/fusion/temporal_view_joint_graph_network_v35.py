"""v35: Temporal View-Joint Graph Network (TVJGN).

Extends the v34 view-joint graph with **temporal edges** that connect each
(view, joint) node to the same node at adjacent timesteps.  This turns the
per-frame skeleton graph into a spatio-temporal graph over (time, view, joint)
nodes, allowing the model to reason about temporal consistency before the final
triangulation head.

The module is intentionally simple: it re-uses the existing graph attention layer
and only changes the graph topology and masking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    CrossViewGraphAttentionLayer,
)
from motionflow_mv.models.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    build_edge_index as _build_single_time_edge_index,
)


_EDGE_TYPE_BONE = 0
_EDGE_TYPE_SYMMETRY = 1
_EDGE_TYPE_CROSS_VIEW = 2
_EDGE_TYPE_SELF = 3
_EDGE_TYPE_TEMPORAL = 4


def _skeleton_for_joints(j: int) -> Tuple[list, list]:
    if j == 17:
        return list(H36M_17_PARENTS), list(H36M_17_SYMMETRY_PAIRS)
    if j == 28:
        return list(MPI_INF_3DHP_28_PARENTS), list(MPI_INF_3DHP_28_SYMMETRY_PAIRS)
    return [-1] + list(range(j - 1)), []


def _build_temporal_edge_index(
    t: int,
    v: int,
    j: int,
    parents: list,
    symmetry_pairs: list,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a spatio-temporal (time, view, joint) graph edge index.

    Edges include the original v34 per-frame edges (bone, symmetry, cross-view,
    self) for each timestep, plus temporal edges connecting the same (view, joint)
    across adjacent timesteps.

    Args:
        t: number of timesteps.
        v: number of views.
        j: number of joints.
        parents: parent index for each joint, -1 for root.
        symmetry_pairs: list of symmetric joint pairs.

    Returns:
        edge_index: (2, E) int64 tensor.
        edge_type: (E,) int64 tensor.
    """
    single_index, single_type = _build_single_time_edge_index(
        parents, symmetry_pairs, v, j, add_self_loops=True
    )
    n_single = single_index.shape[1]

    # Per-timestep edges.
    time_offsets = torch.arange(t, dtype=torch.long) * v * j
    index = single_index.unsqueeze(0) + time_offsets.view(t, 1, 1)  # (T, 2, E_single)
    type_t = single_type.unsqueeze(0).expand(t, -1)  # (T, E_single)

    index_list = [index[i] for i in range(t)]
    type_list = [type_t[i] for i in range(t)]

    # Temporal edges: connect (time, view, joint) <-> (time+1, view, joint).
    node_id = lambda ti, vi, ji: ti * v * j + vi * j + ji  # noqa: E731
    temporal_src, temporal_dst = [], []
    for ti in range(t - 1):
        for vi in range(v):
            for ji in range(j):
                s = node_id(ti, vi, ji)
                d = node_id(ti + 1, vi, ji)
                temporal_src.extend([s, d])
                temporal_dst.extend([d, s])

    if temporal_src:
        temporal_index = torch.stack(
            [torch.tensor(temporal_src, dtype=torch.long), torch.tensor(temporal_dst, dtype=torch.long)],
            dim=0,
        )
        temporal_type = torch.full((temporal_index.shape[1],), _EDGE_TYPE_TEMPORAL, dtype=torch.long)
        index_list.append(temporal_index)
        type_list.append(temporal_type)

    edge_index = torch.cat(index_list, dim=1)
    edge_type = torch.cat(type_list, dim=0)
    return edge_index, edge_type


class TemporalViewJointGraphNetworkV35(nn.Module):
    """Variable-view spatio-temporal view-joint graph attention block.

    Args:
        d: token dimension.
        n_views: maximum number of padded views.
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

        self.layers = nn.ModuleList(
            [CrossViewGraphAttentionLayer(d, n_heads, n_edge_types=5, dropout=dropout) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(d, d)
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        # Gated residual: near-identity at init.
        self.residual_gate = nn.Parameter(torch.tensor(-6.0))

        # Cached graph per (timesteps, joints) signature.
        self._graph_cache: dict = {}

    def _get_graph(self, t: int, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (t, j)
        if key not in self._graph_cache:
            parents, symmetry = _skeleton_for_joints(j)
            edge_index, edge_type = _build_temporal_edge_index(
                t=t,
                v=self.n_views,
                j=j,
                parents=parents,
                symmetry_pairs=symmetry,
            )
            self._graph_cache[key] = (edge_index, edge_type)
        return self._graph_cache[key]

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
        edge_index, edge_type = self._get_graph(T, J)
        edge_index = edge_index.to(tokens.device)
        edge_type = edge_type.to(tokens.device)

        # Zero out padded views so they do not propagate.
        if view_mask is not None:
            tokens = tokens * view_mask[..., None].float()

        # The graph attention layer expects 4-D input (B, V', J', d).  We treat
        # the flattened spatio-temporal token list as a single pseudo-view with
        # T*V*J pseudo-joints, so the layer computes attention over the cached
        # temporal graph correctly.
        x = tokens.view(B, 1, T * V * J, d)

        out = x
        for layer in self.layers:
            out = layer(out, edge_index, edge_type)

        out = self.out_proj(out)
        out = out.view(B, T, V, J, d)

        # Re-apply mask after graph update.
        if view_mask is not None:
            out = out * view_mask[..., None].float()

        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * out
