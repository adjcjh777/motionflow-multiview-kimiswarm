"""Dataset-conditional canonical-skeleton residual refiner.

This module replaces the dense per-joint residual MLP in the anchor model with
a small graph network that predicts pose corrections while conditioning on a
learnable dataset embedding.  The skeleton graph enforces anatomical
consistency by propagating messages along bone and symmetry edges.
"""

from typing import Optional

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


class CanonicalSkeletonResidualRefiner(nn.Module):
    """Graph residual refiner with a per-dataset canonical skeleton prior.

    Parameters
    ----------
    j:
        Number of joints. 17 (H36M / AIST++ layout) or 28 (MPI-INF-3DHP)
        are supported out of the box.
    in_dim:
        Input feature dimension, typically ``d + 3`` (pooled feature +
        raw triangulated 3D pose).
    residual_hidden:
        Hidden dimension of the graph layers.
    num_datasets:
        Number of datasets in the mixed training set (default 3: MPI,
        AIST++, Human3.6M).
    dataset_embed_dim:
        Dimension of the learnable dataset embedding.
    graph_num_layers:
        Number of edge-conditioned graph message-passing layers.
    """

    def __init__(
        self,
        j: int,
        in_dim: int,
        residual_hidden: int = 128,
        num_datasets: int = 3,
        dataset_embed_dim: int = 16,
        graph_num_layers: int = 2,
    ):
        super().__init__()
        self.j = j
        self.in_dim = in_dim
        self.residual_hidden = residual_hidden
        self.num_datasets = num_datasets
        self.dataset_embed_dim = dataset_embed_dim

        if j == 17:
            parents = H36M_17_PARENTS
            symmetry = H36M_17_SYMMETRY_PAIRS
        elif j == 28:
            parents = MPI_INF_3DHP_28_PARENTS
            symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
        else:
            raise NotImplementedError(
                f"CanonicalSkeletonResidualRefiner only supports J=17 or J=28, got {j}"
            )

        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        # Learnable per-dataset embedding and a shared canonical pose offset.
        self.dataset_embed = nn.Embedding(num_datasets, dataset_embed_dim)
        self.canonical_offset = nn.Parameter(torch.zeros(num_datasets, j, 3) * 0.01)

        # Skeleton graph over joints only (n_views=1).
        edge_index, edge_type = build_edge_index(
            parents, symmetry, n_views=1, j=j, add_self_loops=True
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.input_proj = nn.Linear(in_dim + dataset_embed_dim, residual_hidden)
        self.graph = GraphJointRelation(
            d=residual_hidden, n_views=1, num_layers=graph_num_layers
        )
        self.output_proj = nn.Linear(residual_hidden, 3)

    def forward(
        self,
        x: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict residual correction conditioned on dataset identity.

        Parameters
        ----------
        x:
            ``(B, J, in_dim)`` concatenation of pooled feature and raw 3D pose.
        dataset_ids:
            Optional ``(B,)`` long tensor with dataset ids in
            ``[0, num_datasets)``.  If ``None``, all samples are treated as
            dataset 0.

        Returns
        -------
        ``(B, J, 3)`` residual correction.
        """
        B, J, _ = x.shape
        if dataset_ids is None:
            dataset_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        if dataset_ids.numel() == 1 and B > 1:
            dataset_ids = dataset_ids.expand(B)

        emb = self.dataset_embed(dataset_ids)  # (B, E)
        emb = emb.unsqueeze(1).expand(-1, J, -1)  # (B, J, E)

        h = self.input_proj(torch.cat([x, emb], dim=-1))  # (B, J, H)
        h = h.unsqueeze(1)  # (B, 1, J, H)
        h = self.graph(h, self.edge_index, self.edge_type)
        h = h.squeeze(1)  # (B, J, H)

        delta = self.output_proj(h)  # (B, J, 3)
        delta = delta + self.canonical_offset[dataset_ids]  # per-dataset prior
        return delta


def _make_toy_intrinsics(V: int = 4) -> torch.Tensor:
    """Helper for the smoke test."""
    K = torch.eye(3).float().unsqueeze(0).repeat(V, 1, 1)
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    return K


if __name__ == "__main__":
    torch.manual_seed(0)

    for j in (17, 28):
        refiner = CanonicalSkeletonResidualRefiner(
            j=j,
            in_dim=64 + 3,
            residual_hidden=64,
            num_datasets=3,
            dataset_embed_dim=16,
            graph_num_layers=2,
        )
        B = 4
        x = torch.randn(B, j, 67)
        ids = torch.randint(0, 3, (B,))
        delta = refiner(x, ids)
        assert delta.shape == (B, j, 3)
        assert torch.isfinite(delta).all()
        print(f"J={j}: delta mean={delta.mean().item():.4f}, std={delta.std().item():.4f}")

    print("CanonicalSkeletonResidualRefiner smoke test passed")
