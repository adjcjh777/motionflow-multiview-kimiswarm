"""Skeleton-graph residual refiner.

Replaces the dense per-joint residual MLP with a small graph neural network
that propagates pose corrections along bone and symmetry edges.  The module
operates on the joint graph only (no view dimension) and is a drop-in
replacement for ``self.residual_mlp`` in the PP-graph model.
"""

import torch
import torch.nn as nn

from .graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    GraphJointRelation,
    build_edge_index,
)


class SkeletonGraphResidualRefiner(nn.Module):
    """Graph residual refiner operating on the skeleton graph.

    Parameters
    ----------
    j:
        Number of joints (17 or 28 are supported out of the box).
    in_dim:
        Input feature dimension (typically ``d + 3``).
    hidden_dim:
        Hidden dimension of the graph layers.
    num_layers:
        Number of graph message-passing layers.
    """

    def __init__(self, j: int, in_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.j = j
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if j == 17:
            parents = H36M_17_PARENTS
            symmetry = H36M_17_SYMMETRY_PAIRS
        elif j == 28:
            parents = MPI_INF_3DHP_28_PARENTS
            symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
        else:
            raise NotImplementedError(f"SkeletonGraphResidualRefiner does not support J={j}")

        edge_index, edge_type = build_edge_index(parents, symmetry, n_views=1, j=j, add_self_loops=True)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.graph = GraphJointRelation(d=hidden_dim, n_views=1, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        x:
            ``(B, J, in_dim)`` concatenation of pooled feature and raw 3-D pose.

        Returns
        -------
        ``(B, J, 3)`` residual correction.
        """
        # x: (B, J, in_dim)
        h = self.input_proj(x)  # (B, J, hidden_dim)
        # GraphJointRelation expects (B, V, J, d) with V=1.
        h = h.unsqueeze(1)  # (B, 1, J, hidden_dim)
        h = self.graph(h, self.edge_index, self.edge_type)
        h = h.squeeze(1)  # (B, J, hidden_dim)
        return self.output_proj(h)  # (B, J, 3)
