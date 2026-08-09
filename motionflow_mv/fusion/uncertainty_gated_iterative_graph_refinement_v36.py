"""v36: Uncertainty-Gated Iterative Graph Refinement (UGIGR).

Extends the v35 Temporal View-Joint Graph Network by adding:

1. A per-node uncertainty MLP that predicts how reliable each
   ``(time, view, joint)`` token is.
2. Source-gated graph attention: messages from uncertain nodes are
   attenuated, so confident tokens do not get corrupted by noisy neighbors.
3. Iterative refinement with shared weights: the same graph layer is
   unrolled ``n_iters`` times, letting the model progressively denoise the
   feature graph.

The block is identity at initialization (zero output projection and a
near-zero residual gate), so it can be safely stacked on top of v35 without
hurting the baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    CrossViewGraphAttentionLayer,
    _scatter_softmax,
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

    Same topology as v35: per-frame skeleton + cross-view + self loops,
    plus temporal edges connecting the same (view, joint) across adjacent
    timesteps.
    """
    single_index, single_type = _build_single_time_edge_index(
        parents, symmetry_pairs, v, j, add_self_loops=True
    )
    n_single = single_index.shape[1]

    time_offsets = torch.arange(t, dtype=torch.long) * v * j
    index = single_index.unsqueeze(0) + time_offsets.view(t, 1, 1)
    type_t = single_type.unsqueeze(0).expand(t, -1)

    index_list = [index[i] for i in range(t)]
    type_list = [type_t[i] for i in range(t)]

    if t > 1:
        temporal_src, temporal_dst = [], []
        for ti in range(t - 1):
            for vi in range(v):
                for ji_ in range(j):
                    s = ti * v * j + vi * j + ji_
                    d = (ti + 1) * v * j + vi * j + ji_
                    temporal_src.extend([s, d])
                    temporal_dst.extend([d, s])
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


class UncertaintyGatedCrossViewGraphAttentionLayer(CrossViewGraphAttentionLayer):
    """Graph attention layer with optional source-node uncertainty gating.

    Parameters
    ----------
    d:
        Feature dimension.
    n_heads:
        Number of attention heads.
    n_edge_types:
        Number of edge categories.
    dropout:
        Dropout on attention weights.
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        src_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional source gates.

        Args:
            x: (B, V, J, d)
            edge_index: (2, E)
            edge_type: (E,)
            src_gate: optional (B, N, 1) or (B, N, H) gating factor for each
                source node. If None, behaves like the base layer.

        Returns:
            (B, V, J, d)
        """
        B, V, J, _ = x.shape
        N = V * J
        h = x.reshape(B, N, self.d)

        src, dst = edge_index
        E = src.numel()

        q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, N, self.n_heads, self.head_dim)

        q_dst = q[:, dst]
        k_src = k[:, src]
        v_src = v[:, src]

        scores = (q_dst * k_src).sum(dim=-1) / (self.head_dim ** 0.5)
        scores = scores + self.edge_bias(edge_type).unsqueeze(0)

        attn = _scatter_softmax(scores, dst, N)
        attn = self.dropout(attn)

        if src_gate is not None:
            if src_gate.dim() == 3 and src_gate.shape[-1] == 1:
                gate_per_node = src_gate.expand(-1, -1, self.n_heads)
            else:
                gate_per_node = src_gate
            gate_e = gate_per_node.gather(1, src[None, :, None].expand(B, -1, self.n_heads))
            attn = attn * gate_e

        edge_feat = self.edge_emb(edge_type).view(1, E, self.n_heads, self.head_dim)
        v_src = v_src + edge_feat

        out = torch.zeros(B, N, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(1, dst, attn.unsqueeze(-1) * v_src)
        out = out.view(B, N, self.d)

        out = self.out_proj(out)
        out = self.norm(h + out)

        return out.view(B, V, J, self.d)


class UncertaintyGatedIterativeGraphRefinementV36(nn.Module):
    """Uncertainty-gated iterative graph refinement block.

    Args:
        d: token dimension.
        n_views: maximum number of padded views.
        n_layers: number of graph attention layers per iteration.
        n_iters: number of iterative refinement steps (shared weights).
        n_heads: attention heads per layer.
        dropout: dropout on attention weights.
        uncertainty_hidden: hidden dimension of the uncertainty MLP.
        gate_init_bias: initial bias for the uncertainty MLP final layer.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 14,
        n_layers: int = 1,
        n_iters: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        uncertainty_hidden: int = 64,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_layers = n_layers
        self.n_iters = n_iters

        self.layers = nn.ModuleList(
            [UncertaintyGatedCrossViewGraphAttentionLayer(d, n_heads, n_edge_types=5, dropout=dropout)
             for _ in range(n_layers)]
        )

        # Uncertainty MLP: predicts a scalar logit per (time, view, joint) node.
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(d, uncertainty_hidden),
            nn.ReLU(),
            nn.Linear(uncertainty_hidden, 1),
        )
        # Initialize final bias so initial gate is neutral/soft.
        if gate_init_bias != 0.0:
            nn.init.constant_(self.uncertainty_mlp[-1].bias, gate_init_bias)

        # Output projection zeroed at init for identity behaviour.
        self.out_proj = nn.Linear(d, d)
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        # Gated residual: near-zero at init.
        self.residual_gate = nn.Parameter(torch.tensor(-6.0))

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
        reliability_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: (B, T, V, J, d)
            view_mask: optional (B, T, V) bool.
            reliability_gate: optional (B, T, V, J) external per-node reliability
                in [0, 1]. If provided, it is multiplied with the learned
                uncertainty gate so that less reliable nodes send weaker
                messages.

        Returns:
            refined: (B, T, V, J, d)
        """
        B, T, V, J, d = tokens.shape
        edge_index, edge_type = self._get_graph(T, J)
        edge_index = edge_index.to(tokens.device)
        edge_type = edge_type.to(tokens.device)

        # Apply view mask before flattening.
        x = tokens
        if view_mask is not None:
            x = x * view_mask[..., None, None].float()

        # Flatten spatio-temporal nodes: (B, 1, T*V*J, d)
        x = x.view(B, 1, T * V * J, d)

        # Flatten external reliability if provided.
        reliability_flat = None
        if reliability_gate is not None:
            reliability_flat = reliability_gate.view(B, T * V * J, 1)

        # Iterative refinement with shared weights.
        for _ in range(max(self.n_iters, 1)):
            # Recompute uncertainty gates each iteration from current tokens.
            h_nodes = x.view(B, T * V * J, d)
            uncertainty_logits = self.uncertainty_mlp(h_nodes)  # (B, T*V*J, 1)
            # Higher logit -> higher gate -> more message allowed.
            src_gate = torch.sigmoid(uncertainty_logits)

            # v39: couple with external reliability so unreliable nodes
            # propagate less.  Shape (B, T*V*J, 1).
            if reliability_flat is not None:
                src_gate = src_gate * reliability_flat

            for layer in self.layers:
                x = layer(x, edge_index, edge_type, src_gate=src_gate)

        out = self.out_proj(x)
        out = out.view(B, T, V, J, d)

        if view_mask is not None:
            out = out * view_mask[..., None, None].float()

        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * out
