"""Multi-person multi-view association graph for 3-D pose fusion.

Builds a sparse ``(view, person, joint)`` graph across multiple people and views
and runs edge-conditioned message passing.  Designed to be inserted between the
spatio-temporal transformer and the triangulation head of the single-person
ray-attention models.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def build_multiperson_edge_index(
    n_views: int,
    n_persons: int,
    parents: List[int],
    symmetry_pairs: List[Tuple[int, int]],
    add_self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edge index for a ``(view, person, joint)`` graph.

    Edge types are encoded as:
        0 = skeleton (bone + symmetry, intra-person)
        1 = cross-view same-person same-joint
        2 = cross-person same-view same-joint

    Parameters
    ----------
    n_views:
        Number of camera views ``V``.
    n_persons:
        Number of people ``P``.
    parents:
        Parent joint indices for the skeleton; ``-1`` for roots.
    symmetry_pairs:
        List of mirrored left/right joint index pairs.
    add_self_loops:
        Whether to include self-loops on skeleton edges.

    Returns
    -------
    edge_index:
        LongTensor of shape ``(2, E)``.
    edge_type:
        LongTensor of shape ``(E,)`` with values in ``{0, 1, 2}``.
    """
    V, P, J = n_views, n_persons, len(parents)
    edges = []
    edge_type = []

    def node(v: int, p: int, j: int) -> int:
        return v * P * J + p * J + j

    # Intra-person skeleton edges (bone + symmetry) within each view and person.
    for v in range(V):
        for p in range(P):
            for child, parent in enumerate(parents):
                if parent < 0:
                    continue
                a = node(v, p, parent)
                b = node(v, p, child)
                edges.extend([[a, b], [b, a]])
                edge_type.extend([0, 0])
            for lft, rgt in symmetry_pairs:
                a = node(v, p, lft)
                b = node(v, p, rgt)
                edges.extend([[a, b], [b, a]])
                edge_type.extend([0, 0])
            if add_self_loops:
                for j in range(J):
                    a = node(v, p, j)
                    edges.append([a, a])
                    edge_type.append(0)

    # Cross-view same-person same-joint edges.
    for p in range(P):
        for j in range(J):
            for v1 in range(V):
                for v2 in range(v1 + 1, V):
                    a = node(v1, p, j)
                    b = node(v2, p, j)
                    edges.extend([[a, b], [b, a]])
                    edge_type.extend([1, 1])

    # Cross-person same-view same-joint edges.
    for v in range(V):
        for j in range(J):
            for p1 in range(P):
                for p2 in range(p1 + 1, P):
                    a = node(v, p1, j)
                    b = node(v, p2, j)
                    edges.extend([[a, b], [b, a]])
                    edge_type.extend([2, 2])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index, edge_type


class MultiPersonAssociationGraph(nn.Module):
    """Edge-conditioned graph message passing over ``(view, person, joint)`` nodes.

    Input shape:  ``(N, V, P, J, d)``
    Output shape: ``(N, V, P, J, d)``

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of views ``V``.
    n_persons:
        Number of people ``P``.
    num_layers:
        Number of message-passing layers.
    """

    def __init__(self, d: int = 64, n_views: int = 4, n_persons: int = 2, num_layers: int = 2):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_persons = n_persons
        self.num_layers = num_layers

        # One projection per edge type: skeleton, cross-view, cross-person.
        self.edge_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])

        self.edge_attn = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

    def build_edge_index(
        self,
        parents: List[int],
        symmetry_pairs: List[Tuple[int, int]],
        device: torch.device,
    ) -> None:
        """Build and register the static edge index for the current skeleton."""
        edge_index, edge_type = build_multiperson_edge_index(
            self.n_views,
            self.n_persons,
            parents,
            symmetry_pairs,
        )
        self.register_buffer(
            "edge_index",
            edge_index.to(device),
            persistent=False,
        )
        self.register_buffer(
            "edge_type",
            edge_type.to(device),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        parents: List[int],
        symmetry_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Run message passing.

        Parameters
        ----------
        x:
            Input tensor of shape ``(N, V, P, J, d)``.
        parents:
            Skeleton parent indices.
        symmetry_pairs:
            Mirrored joint pairs.

        Returns
        -------
        Refined tensor of shape ``(N, V, P, J, d)``.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5-D input (N, V, P, J, d), got shape {x.shape}")

        N, V, P, J, d = x.shape
        if d != self.d:
            raise ValueError(f"Input feature dim {d} != {self.d}")

        if not hasattr(self, "edge_index") or self.edge_index is None:
            self.build_edge_index(parents, symmetry_pairs, x.device)

        edge_index = self.edge_index
        edge_type = self.edge_type
        nodes_per_graph = V * P * J
        offsets = torch.arange(
            N,
            device=x.device,
            dtype=edge_index.dtype,
        ) * nodes_per_graph
        edge_index = (
            edge_index[:, None, :] + offsets[None, :, None]
        ).reshape(2, -1)
        edge_type = edge_type.repeat(N)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h = x.reshape(-1, self.d)
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

        return h.view(N, V, P, J, self.d)
