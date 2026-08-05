"""Minimal Graph Joint Relation (GJR) prototype.

Run from repo root:
    python docs/swarm_iter_next/design_graph_joint_relation/graph_joint_relation_demo.py
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Skeleton presets
# ---------------------------------------------------------------------------
H36M_17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14, 9]

# Mirrored left/right limb pairs for H36M 17-joint layout.
H36M_SYMMETRY_PAIRS = [(4, 1), (5, 2), (6, 3), (10, 13), (11, 14), (12, 15)]


def build_edge_index(parents, symmetry_pairs, n_views, j, add_self_loops=True):
    """Build an sparse edge list for the (view, joint) graph.

    Returns:
        edge_index: LongTensor of shape (2, E) where E is the number of
                    directed edges. Node id = view * J + joint.
        edge_type: LongTensor of shape (E,) with values in {0, 1, 2}:
                   0 = bone, 1 = symmetry, 2 = cross-view.
    """
    edges = []
    edge_type = []

    # Intra-view skeleton + symmetry edges.
    for v in range(n_views):
        base = v * j
        # bone edges (undirected)
        for child, parent in enumerate(parents):
            if parent < 0:
                continue
            # parent -> child and child -> parent
            edges.extend([[base + parent, base + child], [base + child, base + parent]])
            edge_type.extend([0, 0])
        # symmetry edges (undirected)
        for lft, rgt in symmetry_pairs:
            edges.extend([[base + lft, base + rgt], [base + rgt, base + lft]])
            edge_type.extend([1, 1])
        # self loops
        if add_self_loops:
            for jj in range(j):
                edges.append([base + jj, base + jj])
                edge_type.append(0)  # treat self-loop as bone-type

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

    def __init__(self, d=64, n_views=4, num_layers=3):
        super().__init__()
        self.d = d
        self.num_layers = num_layers

        # Three edge types: bone, symmetry, cross-view.
        self.edge_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])

        # Edge attention: scalar gate from source + target features.
        self.edge_attn = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x:           (B, V, J, d)
            edge_index:  (2, E) LongTensor
            edge_type:   (E,) LongTensor
        Returns:
            (B, V, J, d)
        """
        B, V, J, _ = x.shape
        # Flatten to (B*V*J, d)
        h = x.view(B * V * J, self.d)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        for layer_idx in range(self.num_layers):
            # Source / target node features per edge.
            src = h[src_idx]   # (E, d)
            dst = h[dst_idx]   # (E, d)

            # Edge attention weight per edge.
            attn = torch.sigmoid(self.edge_attn(torch.cat([src, dst], dim=-1))).squeeze(-1)  # (E,)

            # Project source features per edge type.
            projected = torch.zeros_like(src)
            for t in range(3):
                mask = edge_type == t
                if mask.any():
                    projected[mask] = self.edge_proj[t](src[mask])

            # Gate and aggregate messages.
            msg = attn.unsqueeze(-1) * projected  # (E, d)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_idx, msg)

            h = self.norms[layer_idx](h + agg)

        return h.view(B, V, J, self.d)


def main():
    B, V, J, d = 2, 4, 17, 64
    x = torch.randn(B, V, J, d, requires_grad=True)

    edge_index, edge_type = build_edge_index(
        H36M_17_PARENTS, H36M_SYMMETRY_PAIRS, n_views=V, j=J
    )
    print(f"Built graph: {edge_index.shape[1]} directed edges, edge types histogram: {edge_type.bincount().tolist()}")

    gjr = GraphJointRelation(d=d, n_views=V, num_layers=3)
    out = gjr(x, edge_index, edge_type)
    assert out.shape == (B, V, J, d)

    # Backward sanity check.
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    print("Forward/backward sanity check passed; output shape:", out.shape)


if __name__ == "__main__":
    main()
