"""v34: Geometry-aware view-joint graph network.

Extends the content-only ``ViewJointGraphNetworkV34`` by injecting
geometry-derived edge features (epipolar distance + ray-intersection logit)
into the graph attention scores for cross-view edges.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance
from motionflow_mv.fusion.multiview_geometry_fusion_v25 import (
    compute_rays,
    ray_intersection_logit,
)
from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    build_edge_index,
)
from motionflow_mv.models.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)


# ---------------------------------------------------------------------------
# Graph utilities copied from the prototype to avoid relying on internals.
# ---------------------------------------------------------------------------
def _scatter_softmax(
    scores: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------
def _skeleton_for_joints(j: int) -> Tuple[list, list]:
    if j == 17:
        return list(H36M_17_PARENTS), list(H36M_17_SYMMETRY_PAIRS)
    if j == 28:
        return list(MPI_INF_3DHP_28_PARENTS), list(MPI_INF_3DHP_28_SYMMETRY_PAIRS)
    return [-1] + list(range(j - 1)), []


# ---------------------------------------------------------------------------
# Geometry-aware graph attention layer
# ---------------------------------------------------------------------------
class _GeometryAwareGraphAttentionLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")

        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        # Edge geometry feature projection: scalar geometry score -> per-head bias.
        self.geometry_mlp = nn.Sequential(
            nn.Linear(1, d),
            nn.ReLU(),
            nn.Linear(d, n_heads),
        )

        # Edge type bias for non-geometry edges.
        self.edge_type_bias = nn.Embedding(4, n_heads)

        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout) if 0.0 < dropout < 1.0 else nn.Identity()
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        geometry_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, V, J, _ = x.shape
        N = V * J
        h = x.reshape(B, N, self.d)

        src, dst = edge_index
        E = src.numel()

        q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, N, self.n_heads, self.head_dim)
        v = self.k_proj(h).view(B, N, self.n_heads, self.head_dim)

        q_dst = q[:, dst]
        k_src = k[:, src]
        v_src = v[:, src]

        scores = (q_dst * k_src).sum(dim=-1) / (self.head_dim ** 0.5)

        # Content/edge-type bias.
        scores = scores + self.edge_type_bias(edge_type).unsqueeze(0)

        # Geometry bias for cross-view edges.
        if geometry_score is not None:
            geo_bias = self.geometry_mlp(geometry_score.unsqueeze(-1))  # (B, E, n_heads)
            scores = scores + geo_bias

        attn = _scatter_softmax(scores, dst, N)
        attn = self.dropout(attn)

        out = torch.zeros(B, N, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(1, dst, attn.unsqueeze(-1) * v_src)
        out = out.view(B, N, self.d)

        out = self.out_proj(out)
        out = self.norm(h + out)
        return out.view(B, V, J, self.d)


# ---------------------------------------------------------------------------
# Geometry-aware view-joint graph network
# ---------------------------------------------------------------------------
class GeometryViewJointGraphNetworkV34(nn.Module):
    """Variable-view geometry-aware view-joint graph attention block.

    Args:
        d: token dimension.
        n_views: maximum number of padded views (e.g. 14).
        n_layers: number of graph attention layers.
        n_heads: attention heads per layer.
        dropout: dropout on attention weights.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 14,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [_GeometryAwareGraphAttentionLayer(d, n_heads, dropout) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(d, d)
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        self.residual_gate = nn.Parameter(torch.tensor(-6.0))

        # Geometry temperature parameters.
        self.sigma_d = nn.Parameter(torch.tensor(0.5))
        self.sigma_a = nn.Parameter(torch.tensor(0.5))

        self._graph_cache: dict = {}

    def _get_graph(self, j: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if j not in self._graph_cache:
            parents, symmetry = _skeleton_for_joints(j)
            edge_index, edge_type = build_edge_index(parents, symmetry, self.n_views, j, add_self_loops=True)
            self._graph_cache[j] = (edge_index.to(device), edge_type.to(device))
        return self._graph_cache[j]

    def _compute_geometry_score(
        self,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        n_views: int,
        j: int,
    ) -> torch.Tensor:
        """Compute per-edge geometry score for cross-view edges.

        Args:
            points_2d: (B*T, V, J, 2) or (B, T, V, J, 2).
            K: (B*T, V, 3, 3) or (B, T, V, 3, 3).
            R: same as K.
            t: (B*T, V, 3) or (B, T, V, 3).

        Returns:
            score: (B*T, E) scalar for each edge; zero for non-cross-view edges.
        """
        B, T, V, J = points_2d.shape[:4]

        # 4-D versions required by compute_epipolar_distance (treat B*T as batch).
        points_2d_4d = points_2d.reshape(B * T, V, J, 2)
        K_4d = K.reshape(B * T, V, 3, 3)
        R_4d = R.reshape(B * T, V, 3, 3)
        t_4d = t.reshape(B * T, V, 3)

        # Compute full per-joint geometry bias (B*T, V, V, J).
        epi_dist = compute_epipolar_distance(K_4d, R_4d, t_4d, points_2d_4d)

        # compute_rays expects (B, T, V, ...) inputs and produces (B, T, V, V, J).
        centre, direction = compute_rays(points_2d, K, R, t)
        ray_logit = ray_intersection_logit(centre, direction, self.sigma_d, self.sigma_a)
        ray_logit = ray_logit.reshape(B * T, V, V, J)

        geometry_bias = -epi_dist + ray_logit  # (B*T, V, V, J)

        # src/dst are node indices 0..V*J-1. Convert to view/joint.
        src_view = src // J
        src_joint = src % J
        dst_view = dst // J
        dst_joint = dst % J

        # Cross-view edges: same joint, different view.
        cross_view = (src_view != dst_view) & (src_joint == dst_joint)

        score = torch.zeros(B * T, src.numel(), device=points_2d.device, dtype=points_2d.dtype)
        if cross_view.any():
            # Gather geometry_bias[:, src_view, dst_view, src_joint] for cross-view edges.
            sv = src_view[cross_view]
            dv = dst_view[cross_view]
            sj = src_joint[cross_view]
            gathered = geometry_bias[:, sv, dv, sj]  # (B*T, n_cross)
            score[:, cross_view] = gathered
        return score

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        points_2d: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            view_mask: optional (B, T, V) bool.
            points_2d: optional (B, T, V, J, 2). Required for geometry.
            K: optional (B, T, V, 3, 3). Required for geometry.
            R: optional (B, T, V, 3, 3). Required for geometry.
            t: optional (B, T, V, 3). Required for geometry.
        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        edge_index, edge_type = self._get_graph(J, tokens.device)

        x = tokens.reshape(B * T, V, J, d)

        if view_mask is not None:
            x = x * view_mask.reshape(B * T, V, 1, 1).float()

        geometry_score = None
        if points_2d is not None and K is not None and R is not None and t is not None:
            geometry_score = self._compute_geometry_score(
                points_2d=points_2d,
                K=K,
                R=R,
                t=t,
                src=edge_index[0],
                dst=edge_index[1],
                n_views=V,
                j=J,
            )

        out = x
        for layer in self.layers:
            out = layer(out, edge_index, edge_type, geometry_score=geometry_score)

        out = self.out_proj(out)

        if view_mask is not None:
            out = out * view_mask.reshape(B * T, V, 1, 1).float()

        out = out.view(B, T, V, J, d)
        gate = torch.sigmoid(self.residual_gate)
        return tokens + gate * out
