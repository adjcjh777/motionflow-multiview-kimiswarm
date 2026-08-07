"""Graph-joint attention v2.

A sparse, skeleton-aware attention module over the ``(view, joint)`` product
graph.  Compared with the dense transformer-based ``joint_attn`` and the
original ``GraphJointRelation`` block, this version:

* uses multi-head dot-product attention instead of a single MLP gate,
* introduces explicit edge-type embeddings and per-head scalar biases for
  bone / symmetry / cross-view / self-loop edges,
* normalises attention weights per destination node and head (scatter softmax),
* adds an optional point-wise FFN and dropout inside each layer,
* stays fully batched over the edge list so it works for any view count.

Input shape:  ``(B, V, J, d)``
Output shape: ``(B, V, J, d)``
"""

from typing import List, Tuple

import math
import torch
import torch.nn as nn

from .graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)


EDGE_TYPE_BONE = 0
EDGE_TYPE_SYMMETRY = 1
EDGE_TYPE_CROSS_VIEW = 2
EDGE_TYPE_SELF = 3


def build_graph_joint_edge_index(
    parents: List[int],
    symmetry_pairs: List[Tuple[int, int]],
    n_views: int,
    j: int,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a directed (view, joint) graph for graph-joint attention v2.

    Edge types:
        * ``0`` -- bone (parent <-> child) within each view
        * ``1`` -- symmetry (left/right mirror pairs) within each view
        * ``2`` -- cross-view same-joint edges
        * ``3`` -- self-loop identity edges

    Parameters
    ----------
    parents:
        Parent index for each joint, ``-1`` for the root.
    symmetry_pairs:
        List of symmetric joint index pairs.
    n_views:
        Number of camera views.
    j:
        Number of joints.
    add_self_loops:
        Whether to include identity edges.

    Returns
    -------
    edge_index:
        LongTensor of shape ``(2, E)``.
    edge_type:
        LongTensor of shape ``(E,)`` with values in ``{0,1,2,3}``.
    """
    edges: List[List[int]] = []
    edge_type: List[int] = []

    # Intra-view bone and symmetry edges.
    for v in range(n_views):
        base = v * j
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            edges.extend([[base + parent, base + child], [base + child, base + parent]])
            edge_type.extend([EDGE_TYPE_BONE, EDGE_TYPE_BONE])

        for left, right in symmetry_pairs:
            edges.extend([[base + left, base + right], [base + right, base + left]])
            edge_type.extend([EDGE_TYPE_SYMMETRY, EDGE_TYPE_SYMMETRY])

        if add_self_loops:
            for jj in range(j):
                edges.append([base + jj, base + jj])
                edge_type.append(EDGE_TYPE_SELF)

    # Cross-view same-joint edges.
    for v1 in range(n_views):
        for v2 in range(v1 + 1, n_views):
            for jj in range(j):
                a = v1 * j + jj
                b = v2 * j + jj
                edges.extend([[a, b], [b, a]])
                edge_type.extend([EDGE_TYPE_CROSS_VIEW, EDGE_TYPE_CROSS_VIEW])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    return edge_index, edge_type_t


def _scatter_softmax(
    logits: torch.Tensor,
    dst_idx: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Stable scatter softmax over incoming edges for each destination/head.

    Parameters
    ----------
    logits:
        ``(B, E, H)`` raw attention scores.
    dst_idx:
        ``(E,)`` destination node for each edge.
    num_nodes:
        Total number of nodes ``V * J``.

    Returns
    -------
    attn:
        ``(B, E, H)`` normalised attention weights.
    """
    B, E, H = logits.shape
    device = logits.device
    dtype = logits.dtype

    # Broadcast destination indices to (B, E, H).
    dst_idx_b = dst_idx.unsqueeze(0).unsqueeze(-1).expand(B, E, H)

    # Numerical stabilisation: subtract per-destination max per batch.
    max_per_node = torch.full((B, num_nodes, H), -float("inf"), device=device, dtype=dtype)
    try:
        max_per_node.scatter_reduce_(
            1,
            dst_idx_b,
            logits,
            reduce="amax",
            include_self=False,
        )
    except (RuntimeError, AttributeError):
        pass

    max_per_edge = max_per_node.gather(1, dst_idx_b)  # (B, E, H)
    logits = logits - max_per_edge

    exp_logits = torch.exp(logits)
    sum_per_node = torch.zeros((B, num_nodes, H), device=device, dtype=dtype)
    sum_per_node.scatter_add_(
        1,
        dst_idx_b,
        exp_logits,
    )
    sum_per_edge = sum_per_node.gather(1, dst_idx_b)  # (B, E, H)
    return exp_logits / (sum_per_edge + 1e-12)


class GraphJointAttentionLayer(nn.Module):
    """Single graph-joint attention layer.

    Parameters
    ----------
    d:
        Node feature dimension. Must be divisible by ``n_heads``.
    n_heads:
        Number of attention heads.
    n_edge_types:
        Number of distinct edge categories (default 4).
    dropout:
        Dropout applied to normalised attention weights.
    ffn_hidden:
        Hidden dimension of the optional point-wise FFN.  If ``None`` or
        non-positive, no FFN is used.
    """

    def __init__(
        self,
        d: int = 64,
        n_heads: int = 4,
        n_edge_types: int = 4,
        dropout: float = 0.0,
        ffn_hidden: int = 0,
    ):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")

        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # Per-edge-type embedding added to source values, and per-head bias for scores.
        self.edge_emb = nn.Embedding(n_edge_types, d)
        self.edge_bias = nn.Embedding(n_edge_types, n_heads)

        self.attn_dropout = nn.Dropout(dropout) if 0.0 < dropout < 1.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d)

        self.use_ffn = ffn_hidden is not None and ffn_hidden > 0
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d, ffn_hidden),
                nn.GELU(),
                nn.Linear(ffn_hidden, d),
            )
            self.norm2 = nn.LayerNorm(d)

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

        # Multi-head projections.
        q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, N, self.n_heads, self.head_dim)

        # Gather edge endpoints.
        q_dst = q[:, dst]  # (B, E, H, hd)
        k_src = k[:, src]
        v_src = v[:, src]

        # Scaled dot-product attention with per-edge-type per-head bias.
        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # (B, E, H)
        scores = scores + self.edge_bias(edge_type).unsqueeze(0)  # (B, E, H)

        attn = _scatter_softmax(scores, dst, N)  # (B, E, H)
        attn = self.attn_dropout(attn)

        # Add edge-type embeddings to source values.
        edge_feat = self.edge_emb(edge_type).view(1, E, self.n_heads, self.head_dim)
        v_src = v_src + edge_feat

        # Aggregate weighted messages to each destination node.
        out = torch.zeros(B, N, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(1, dst, attn.unsqueeze(-1) * v_src)
        out = out.view(B, N, self.d)

        out = self.out_proj(out)
        h = self.norm1(h + out)

        if self.use_ffn:
            h = self.norm2(h + self.ffn(h))

        return h.view(B, V, J, self.d)


class GraphJointAttentionV2(nn.Module):
    """Stack of graph-joint attention layers.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views (kept for API compatibility).
    n_layers:
        Number of stacked graph attention layers.
    n_heads:
        Attention heads per layer.
    n_edge_types:
        Number of edge categories passed to each layer.
    dropout:
        Attention dropout.
    ffn_hidden:
        Hidden dimension of the optional point-wise FFN in each layer.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        n_layers: int = 2,
        n_heads: int = 4,
        n_edge_types: int = 4,
        dropout: float = 0.0,
        ffn_hidden: int = 0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            GraphJointAttentionLayer(d, n_heads, n_edge_types, dropout, ffn_hidden)
            for _ in range(n_layers)
        )

        # Default edge buffers; callers should normally supply an edge list.
        self.register_buffer("edge_index", torch.zeros((2, 1), dtype=torch.long))
        self.register_buffer("edge_type", torch.zeros((1,), dtype=torch.long))

    def build_edge_index(
        self,
        j: int,
        parents: List[int],
        symmetry_pairs: List[Tuple[int, int]],
        add_self_loops: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build and cache the (view, joint) graph for the current skeleton.

        Parameters
        ----------
        j:
            Number of joints.
        parents:
            Parent index list, ``-1`` for the root.
        symmetry_pairs:
            List of symmetric joint index pairs.
        add_self_loops:
            Include identity edges.

        Returns
        -------
        edge_index, edge_type tensors; also stored as buffers.
        """
        edge_index, edge_type = build_graph_joint_edge_index(
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

        Parameters
        ----------
        x:
            Input features of shape ``(B, V, J, d)``.
        edge_index:
            Optional edge list of shape ``(2, E)``.  If omitted, the cached
            buffers are used.
        edge_type:
            Optional edge type tensor of shape ``(E,)``.

        Returns
        -------
        out:
            Refined features of shape ``(B, V, J, d)``.
        """
        if edge_index is None:
            edge_index = self.edge_index.to(x.device)
        if edge_type is None:
            edge_type = self.edge_type.to(x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x
