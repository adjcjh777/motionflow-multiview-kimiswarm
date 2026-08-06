"""Kinematic-chain graph convolutional refiner (KC-GCR).

Operates directly on the triangulated 3-D skeleton, propagating corrections
along anatomical bone and symmetry edges with edge-conditioned graph
convolutions.  The module is a drop-in final refiner: it keeps the existing
attention-based fusion and residual head untouched, then performs one more
skeleton-aware refinement step on the output pose space.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    build_edge_index,
)


def _get_skeleton(j: int) -> Tuple[list[int], list[Tuple[int, int]]]:
    """Return (parents, symmetry_pairs) for supported skeletons."""
    if j == 17:
        return H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS
    if j == 28:
        return MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS
    raise NotImplementedError(f"KinematicChainGraphRefiner does not support J={j}")


class KinematicChainGraphRefiner(nn.Module):
    """Skeleton-aware 3-D pose refiner using graph convolutions on the kinematic chain.

    Parameters
    ----------
    j:
        Number of joints (17 or 28 supported).
    hidden_dim:
        Width of the graph feature space.
    num_layers:
        Number of edge-conditioned message-passing layers.
    share_weights:
        If True, all graph layers share the same parameters.
    """

    def __init__(
        self,
        j: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 2,
        share_weights: bool = True,
    ):
        super().__init__()
        self.j = j
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.share_weights = share_weights

        parents, symmetry = _get_skeleton(j)
        edge_index, edge_type = build_edge_index(
            parents, symmetry, n_views=1, j=j, add_self_loops=True
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        # 3D coordinate -> feature
        self.input_proj = nn.Linear(3, hidden_dim)

        # Edge-conditioned graph convolution.
        # We distinguish bone, symmetry, and self-loop edges.
        self.edge_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Feature -> 3D correction
        self.output_proj = nn.Linear(hidden_dim, 3)

    def _message_passing(self, h: torch.Tensor) -> torch.Tensor:
        """One edge-conditioned graph convolution step.

        Args:
            h: ``(B, J, hidden_dim)`` node features.

        Returns:
            Updated ``(B, J, hidden_dim)`` node features.
        """
        B, J, D = h.shape
        # Flatten to (B*J, D)
        flat = h.reshape(B * J, D)
        src = flat[self.edge_index[0]]
        dst = flat[self.edge_index[1]]

        # Edge attention weights.
        attn = torch.sigmoid(self.edge_attention(torch.cat([src, dst], dim=-1))).squeeze(-1)

        # Project source features per edge type.
        projected = torch.zeros_like(src)
        for t in range(len(self.edge_projections)):
            mask = self.edge_type == t
            if mask.any():
                projected[mask] = self.edge_projections[t](src[mask])

        msg = attn.unsqueeze(-1) * projected

        # Aggregate messages to destination nodes.
        agg = torch.zeros_like(flat)
        agg.index_add_(0, self.edge_index[1], msg)

        return agg.reshape(B, J, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a 3-D skeleton.

        Args:
            x: ``(B, J, 3)`` world-coordinate joints.

        Returns:
            ``(B, J, 3)`` refined joints (input + residual correction).
        """
        h = self.input_proj(x)
        for i in range(self.num_layers):
            msg = self._message_passing(h)
            h = self.layer_norms[i](h + msg)
        delta = self.output_proj(h)
        return x + delta


class KinematicChainGraphRefinerTemporal(nn.Module):
    """Temporal-aware wrapper that applies ``KinematicChainGraphRefiner`` per frame.

    Useful when the upstream model already outputs a temporal sequence
    ``(B, T, J, 3)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.refiner = KinematicChainGraphRefiner(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine ``(B, T, J, 3)`` or ``(B, J, 3)`` skeletons."""
        if x.dim() == 4:
            B, T, J, C = x.shape
            x_flat = x.reshape(B * T, J, C)
            out = self.refiner(x_flat)
            return out.reshape(B, T, J, C)
        return self.refiner(x)
