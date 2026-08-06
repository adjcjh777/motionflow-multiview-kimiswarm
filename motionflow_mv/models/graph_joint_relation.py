"""Graph Joint Relation (GJR) module for skeleton-aware multi-view fusion.

Implements a sparse graph attention block over the (view, joint) skeleton
graph.  Edges encode bone (parent-child), left/right symmetry, and
same-joint cross-view relationships.  The module is intended as a
drop-in replacement for dense joint-level self-attention.
"""

from typing import List, Tuple

import math
import torch
import torch.nn as nn


# Human3.6M 17-joint skeleton (parent index, -1 for root).
H36M_17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 16]

# Mirrored left/right limb pairs for the H36M 17-joint layout.
H36M_17_SYMMETRY_PAIRS = [(4, 1), (5, 2), (6, 3), (10, 13), (11, 14), (12, 15)]

# MPI-INF-3DHP 28-joint skeleton (parent index, -1 for root).
MPI_INF_3DHP_28_PARENTS = [
    2,  # 0  spine3        -> 2  spine2
    0,  # 1  spine4        -> 0  spine3
    3,  # 2  spine2        -> 3  spine
    4,  # 3  spine         -> 4  pelvis
    -1, # 4  pelvis        (root)
    1,  # 5  neck          -> 1  spine4
    5,  # 6  head          -> 5  neck
    6,  # 7  head_top      -> 6  head
    5,  # 8  left_clavicle -> 5  neck
    8,  # 9  left_shoulder -> 8  left_clavicle
    9,  # 10 left_elbow    -> 9  left_shoulder
    10, # 11 left_wrist    -> 10 left_elbow
    11, # 12 left_hand     -> 11 left_wrist
    5,  # 13 right_clavicle-> 5  neck
    13, # 14 right_shoulder-> 13 right_clavicle
    14, # 15 right_elbow   -> 14 right_shoulder
    15, # 16 right_wrist   -> 15 right_elbow
    16, # 17 right_hand    -> 16 right_wrist
    4,  # 18 left_hip      -> 4  pelvis
    18, # 19 left_knee     -> 18 left_hip
    19, # 20 left_ankle    -> 19 left_knee
    20, # 21 left_foot     -> 20 left_ankle
    21, # 22 left_toe      -> 21 left_foot
    4,  # 23 right_hip     -> 4  pelvis
    23, # 24 right_knee    -> 23 right_hip
    24, # 25 right_ankle   -> 24 right_knee
    25, # 26 right_foot    -> 25 right_ankle
    26, # 27 right_toe     -> 26 right_foot
]

MPI_INF_3DHP_28_SYMMETRY_PAIRS = [
    (8, 13),
    (9, 14),
    (10, 15),
    (11, 16),
    (12, 17),
    (18, 23),
    (19, 24),
    (20, 25),
    (21, 26),
    (22, 27),
]


def build_edge_index(
    parents: List[int],
    symmetry_pairs: List[Tuple[int, int]],
    n_views: int,
    j: int,
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a sparse edge list for the (view, joint) graph.

    Returns
    -------
    edge_index:
        LongTensor of shape ``(2, E)`` for directed edges.
        Node id = ``view * J + joint``.
    edge_type:
        LongTensor of shape ``(E,)`` with values in ``{0, 1, 2}``:
        ``0 = bone``, ``1 = symmetry``, ``2 = cross-view``.
    """
    edges = []
    edge_type = []

    # Intra-view skeleton + symmetry edges.
    for v in range(n_views):
        base = v * j
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            edges.extend([[base + parent, base + child], [base + child, base + parent]])
            edge_type.extend([0, 0])
        for lft, rgt in symmetry_pairs:
            edges.extend([[base + lft, base + rgt], [base + rgt, base + lft]])
            edge_type.extend([1, 1])
        if add_self_loops:
            for jj in range(j):
                edges.append([base + jj, base + jj])
                edge_type.append(0)

    # Cross-view same-joint edges.
    for v1 in range(n_views):
        for v2 in range(v1 + 1, n_views):
            for jj in range(j):
                a = v1 * j + jj
                b = v2 * j + jj
                edges.extend([[a, b], [b, a]])
                edge_type.extend([2, 2])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index, edge_type


class GraphJointRelation(nn.Module):
    """Multi-head graph attention over the (view, joint) skeleton graph.

    Input shape:  ``(B, V, J, d)``
    Output shape: ``(B, V, J, d)``

    Parameters
    ----------
    in_dim:
        Node feature dimension. Must be divisible by ``num_heads``.
    n_views:
        Number of views (kept for API compatibility).
    num_layers:
        Number of message-passing layers.
    num_heads:
        Number of attention heads.
    """

    def __init__(self, in_dim: int = 64, n_views: int = 4, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})")

        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        # Multi-head Q/K/V projections.
        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.o_proj = nn.Linear(in_dim, in_dim)

        # Edge-type embedding (bone / symmetry / cross-view) biases attention.
        self.edge_type_embed = nn.Embedding(3, in_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_heads),
        )

        self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(num_layers)])

    def _softmax_by_dst(self, logits: torch.Tensor, dst_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Stable softmax over incoming edges for each destination node and head."""
        # logits: (E, H)
        H = logits.size(1)
        try:
            max_per_node = torch.full((num_nodes, H), -float("inf"), device=logits.device, dtype=logits.dtype)
            max_per_node.scatter_reduce_(
                0, dst_idx.unsqueeze(-1).expand(-1, H), logits, reduce="amax", include_self=False
            )
            logits = logits - max_per_node[dst_idx]
        except (RuntimeError, AttributeError):
            # Fallback for older PyTorch builds without scatter_reduce amax.
            pass

        exp_logits = torch.exp(logits)
        sum_exp = torch.zeros(num_nodes, H, device=logits.device, dtype=logits.dtype)
        sum_exp.index_add_(0, dst_idx, exp_logits)
        attn = exp_logits / (sum_exp[dst_idx] + 1e-12)
        return attn

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Run graph attention message passing.

        Args
        ----
        x:
            Input features of shape ``(B, V, J, d)``.
        edge_index:
            Directed edge list of shape ``(2, E)``.
        edge_type:
            Edge type labels of shape ``(E,)``.
        return_intermediates:
            If ``True``, also return a list of per-layer output tensors.

        Returns
        -------
        out:
            Refined features of shape ``(B, V, J, d)``.
        intermediates (optional):
            List of per-layer output tensors, each ``(B, V, J, d)``.
        """
        B, V, J, _ = x.shape
        h = x.view(B * V * J, self.in_dim)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        edge_type_emb = self.edge_type_embed(edge_type)

        intermediates = []
        for layer_idx, norm in enumerate(self.norms):
            q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
            v = self.v_proj(h).view(-1, self.num_heads, self.head_dim)

            q_dst = q[dst_idx]  # (E, H, hd)
            k_src = k[src_idx]  # (E, H, hd)
            v_src = v[src_idx]  # (E, H, hd)

            # Dot-product attention scaled by sqrt(d).
            logits = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # (E, H)

            # Edge-type and node-pair bias.
            mlp_input = torch.cat([h[src_idx], h[dst_idx], edge_type_emb], dim=-1)  # (E, 3d)
            logits = logits + self.attn_mlp(mlp_input)

            # Softmax over incoming edges per destination/head.
            attn = self._softmax_by_dst(logits, dst_idx, h.size(0))  # (E, H)

            # Aggregate gated messages.
            msg = attn.unsqueeze(-1) * v_src  # (E, H, hd)
            out = torch.zeros_like(q)  # (N, H, hd)
            out.index_add_(0, dst_idx, msg)
            out = out.view(-1, self.in_dim)

            # Output projection and residual.
            out = self.o_proj(out)
            h = norm(h + out)

            if return_intermediates:
                intermediates.append(h.view(B, V, J, self.in_dim).detach())

        out = h.view(B, V, J, self.in_dim)
        if return_intermediates:
            return out, intermediates
        return out
