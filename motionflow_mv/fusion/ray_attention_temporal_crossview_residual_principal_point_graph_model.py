"""Skeleton-graph drop-in replacement for dense joint-level self-attention.

Extends the best ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
model by replacing the ``joint_attn`` layers in the per-frame encoder with
``GraphJointRelation``.  The graph operates over the (view, joint) skeleton
graph and is therefore anatomy-aware while retaining cross-view edges.

This is a minimal warm-start skeleton: it keeps every other component of the
best PP model (PP correction, residual refinement, weight head) unchanged so
that only the new graph-attention block needs to be validated.
"""

import torch
import torch.nn as nn

from .graph_joint_relation import (
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    GraphJointRelation,
    build_edge_index,
)
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointGraph(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP model with skeleton graph attention.

    Parameters
    ----------
    graph_num_layers:
        Number of ``GraphJointRelation`` message-passing layers used in place
        of the dense ``joint_attn`` transformer layers.  Default matches the
        base model's ``n_joint_layers`` (1).
    graph_share_weights:
        If ``True`` (default) all graph layers share the same
        ``GraphJointRelation`` instance to keep parameter count low.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        graph_num_layers: int = 1,
        graph_share_weights: bool = True,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=return_visibility,
        )

        if graph_num_layers < 1:
            raise ValueError("graph_num_layers must be >= 1")

        self.graph_num_layers = graph_num_layers
        self.graph_share_weights = graph_share_weights

        # Build the static (view, joint) skeleton graph once.
        # We use the MPI-INF-3DHP skeleton for the 28-joint case and fall back
        # to H36M for 17-joint inputs.  Other skeleton layouts can be injected
        # later by overriding _build_edge_index.
        self.register_buffer(
            "edge_index",
            torch.zeros((2, 1), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "edge_type",
            torch.zeros((1,), dtype=torch.long),
            persistent=False,
        )

        # Replace dense joint attention with the graph module.
        graph_module = GraphJointRelation(d=d, n_views=n_views, num_layers=1)
        if graph_share_weights:
            self.joint_graph = nn.ModuleList([graph_module] * graph_num_layers)
        else:
            self.joint_graph = nn.ModuleList(
                [GraphJointRelation(d=d, n_views=n_views, num_layers=1) for _ in range(graph_num_layers)]
            )

    def _build_edge_index(self, n_views: int, j: int, device: torch.device):
        """Build (or refresh) edge index for the current skeleton and view count."""
        if j == 28:
            parents = MPI_INF_3DHP_28_PARENTS
            symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
        elif j == 17:
            from .graph_joint_relation import H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS

            parents = H36M_17_PARENTS
            symmetry = H36M_17_SYMMETRY_PAIRS
        else:
            raise NotImplementedError(
                f"GraphJointRelation skeleton layout for J={j} is not yet implemented."
            )
        edge_index, edge_type = build_edge_index(parents, symmetry, n_views, j, add_self_loops=True)
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)

    def _extract_frame_features(self, x, K, R, t) -> torch.Tensor:
        """Per-frame encoder with graph attention replacing dense joint attention."""
        N, V, J, _ = x.shape
        device = x.device

        # Lazy (re-)build edge index when shape changes.
        if self.edge_index.numel() == 2 or self.edge_index.shape[1] < 2:
            self._build_edge_index(V, J, device)

        points_2d = x[..., :2]

        from .ray_attention_model import _compute_rays

        rays = _compute_rays(points_2d, K, R, t)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)

        obs_emb = self.obs_embed(x)
        ray_emb = self.ray_embed(ray_input)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)

        camera_feat = torch.cat([K.view(N, V, -1), R.view(N, V, -1), t.view(N, V, -1)], dim=-1)
        camera_emb = self.camera_embed_mlp(camera_feat)
        camera_emb = camera_emb[:, :, None, :].expand(N, V, J, self.d)
        feat = feat + camera_emb

        # View-level attention (unchanged).
        feat_v = feat.permute(0, 2, 1, 3).reshape(N * J, V, self.d)
        attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
        feat_v = self.view_norm1(feat_v + attn_out)
        feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))
        feat_v = feat_v.view(N, J, V, self.d)

        # Joint-level **graph** attention (replaces dense transformer layers).
        feat_j = feat_v.permute(0, 2, 1, 3).reshape(N, V, J, self.d).contiguous()
        for graph_layer in self.joint_graph:
            feat_j = graph_layer(feat_j, self.edge_index, self.edge_type)
        feat_j = feat_j.view(N, V, J, self.d)

        return feat_j
