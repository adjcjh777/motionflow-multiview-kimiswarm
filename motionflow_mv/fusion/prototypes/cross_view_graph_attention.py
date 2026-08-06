"""Cross-view graph attention fusion module prototype.

Operates over a sparse ``(view, joint)`` graph whose edges are:

* **bone** – parent/child skeleton links within each view
* **symmetry** – left/right symmetric joints within each view
* **cross-view** – the same joint across different views
* **self** – identity edges (residual connection handled externally)

The module computes multi-head attention over incoming neighbors for each
``(view, joint)`` node, conditioned on edge type, and can be used as a
drop-in replacement for the dense transformer-based joint attention in the
ray-attention fusion models.

This file is intentionally self-contained and lives in
``motionflow_mv/fusion/prototypes/`` so it can be iterated on without touching
running experiments or other model code.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from ..graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    build_edge_index,
)


def _scatter_softmax(
    scores: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Softmax over graph destinations using index operations.

    Args:
        scores: (B, E, H) raw attention scores per edge/head.
        dst: (E,) destination node index for each edge.
        n_nodes: total number of destination nodes.

    Returns:
        attn: (B, E, H) normalized attention weights.
    """
    B, E, H = scores.shape
    device = scores.device
    dtype = scores.dtype

    max_score = torch.full((B, n_nodes, H), float("-inf"), device=device, dtype=dtype)
    max_score = max_score.index_reduce_(1, dst, scores, reduce="amax", include_self=True)
    max_per_edge = max_score.gather(1, dst[None, :, None].expand(B, -1, H))

    exp_scores = torch.exp(scores - max_per_edge)
    sum_exp = torch.zeros(B, n_nodes, H, device=device, dtype=dtype)
    sum_exp.index_add_(1, dst, exp_scores)
    sum_per_edge = sum_exp.gather(1, dst[None, :, None].expand(B, -1, H))

    return exp_scores / (sum_per_edge + 1e-8)


class CrossViewGraphAttentionLayer(nn.Module):
    """Single graph attention layer over the (view, joint) skeleton graph.

    Input shape:  (B, V, J, d)
    Output shape: (B, V, J, d)

    Parameters
    ----------
    d:
        Node feature dimension.
    n_heads:
        Number of attention heads; ``d`` must be divisible by ``n_heads``.
    n_edge_types:
        Number of edge categories (default 4: bone, symmetry, cross-view, self).
    dropout:
        Dropout applied to attention weights.
    """

    def __init__(self, d: int, n_heads: int = 4, n_edge_types: int = 4, dropout: float = 0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")

        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        # Per-edge feature bias for values and per-head scalar bias for scores.
        self.edge_emb = nn.Embedding(n_edge_types, d)
        self.edge_bias = nn.Embedding(n_edge_types, n_heads)

        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout) if 0.0 < dropout < 1.0 else nn.Identity()
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        B, V, J, _ = x.shape
        N = V * J
        h = x.reshape(B, N, self.d)

        src, dst = edge_index
        E = src.numel()

        # Multi-head projections for all nodes.
        q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, N, self.n_heads, self.head_dim)

        # Gather source/destination features for each edge.
        q_dst = q[:, dst]  # (B, E, H, head_dim)
        k_src = k[:, src]
        v_src = v[:, src]

        # Scaled dot-product attention, with per-edge-type per-head bias.
        scores = (q_dst * k_src).sum(dim=-1) / (self.head_dim ** 0.5)  # (B, E, H)
        scores = scores + self.edge_bias(edge_type).unsqueeze(0)  # (B, E, H)

        attn = _scatter_softmax(scores, dst, N)  # (B, E, H)
        attn = self.dropout(attn)

        # Add edge-type embeddings to the source value features.
        edge_feat = self.edge_emb(edge_type).view(1, E, self.n_heads, self.head_dim)
        v_src = v_src + edge_feat

        # Aggregate weighted messages per destination node.
        out = torch.zeros(B, N, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(1, dst, attn.unsqueeze(-1) * v_src)
        out = out.view(B, N, self.d)

        out = self.out_proj(out)
        out = self.norm(h + out)

        return out.view(B, V, J, self.d)


class CrossViewGraphAttention(nn.Module):
    """Stack of cross-view graph attention layers.

    Builds a static ``(view, joint)`` graph from a skeleton definition and
    runs graph attention over it.  The graph is constructed once and cached as
    a buffer.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views.
    n_layers:
        Number of stacked graph attention layers.
    n_heads:
        Attention heads per layer.
    n_edge_types:
        Edge type count passed to each layer.
    dropout:
        Attention dropout.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        n_layers: int = 2,
        n_heads: int = 4,
        n_edge_types: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            CrossViewGraphAttentionLayer(d, n_heads, n_edge_types, dropout)
            for _ in range(n_layers)
        )

        self.register_buffer("edge_index", torch.zeros((2, 1), dtype=torch.long))
        self.register_buffer("edge_type", torch.zeros((1,), dtype=torch.long))

    def build_edge_index(
        self,
        j: int,
        parents: List[int],
        symmetry_pairs: List[Tuple[int, int]],
        add_self_loops: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the (view, joint) graph for the current skeleton and view count.

        Args:
            j: number of joints.
            parents: parent index for each joint, -1 for root.
            symmetry_pairs: list of symmetric joint index pairs.
            add_self_loops: include identity edges.

        Returns:
            edge_index, edge_type tensors, also stored as buffers.
        """
        edge_index, edge_type = build_edge_index(
            parents, symmetry_pairs, self.n_views, j, add_self_loops
        )
        self.edge_index = edge_index
        self.edge_type = edge_type
        return edge_index, edge_type

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, V, J, d)
            edge_index: optional edge list of shape (2, E). If omitted, uses
                the cached graph buffers.
            edge_type: optional edge type tensor of shape (E,). Must be
                provided if ``edge_index`` is provided.

        Returns:
            (B, V, J, d)
        """
        if edge_index is None:
            edge_index = self.edge_index.to(x.device)
        if edge_type is None:
            edge_type = self.edge_type.to(x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x
