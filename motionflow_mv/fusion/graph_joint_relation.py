"""Graph Joint Relation (GJR) module for skeleton-aware multi-view fusion.

Builds a sparse (view, joint) graph with bone, symmetry and cross-view edges,
and runs edge-conditioned message passing.  This can replace the dense
transformer-based ``joint_attn`` in the ray-attention models.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


# Human3.6M 17-joint skeleton (parent index, -1 for root).
H36M_17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 16]

# Mirrored left/right limb pairs for H36M 17-joint layout.
H36M_17_SYMMETRY_PAIRS = [(4, 1), (5, 2), (6, 3), (10, 13), (11, 14), (12, 15)]


# MPI-INF-3DHP 28-joint skeleton.
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
        LongTensor of shape (2, E) for directed edges. Node id = view * J + joint.
    edge_type:
        LongTensor of shape (E,) with values in {0, 1, 2}:
        0 = bone, 1 = symmetry, 2 = cross-view.
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
    """Edge-conditioned graph message passing over a (view, joint) skeleton graph.

    Input shape:  (B, V, J, d)
    Output shape: (B, V, J, d)
    """

    def __init__(self, d: int = 64, n_views: int = 4, num_layers: int = 3):
        super().__init__()
        self.d = d
        self.num_layers = num_layers

        # Three edge types: bone, symmetry, cross-view.
        self.edge_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])

        self.edge_attn = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        B, V, J, _ = x.shape
        h = x.view(B * V * J, self.d)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        for layer_idx in range(self.num_layers):
            src = h[src_idx]
            dst = h[dst_idx]

            attn = torch.sigmoid(self.edge_attn(torch.cat([src, dst], dim=-1))).squeeze(-1)

            projected = torch.zeros_like(src)
            for t in range(3):
                mask = edge_type == t
                if mask.any():
                    projected[mask] = self.edge_proj[t](src[mask])

            msg = attn.unsqueeze(-1) * projected
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_idx, msg)

            h = self.norms[layer_idx](h + agg)

        return h.view(B, V, J, self.d)
