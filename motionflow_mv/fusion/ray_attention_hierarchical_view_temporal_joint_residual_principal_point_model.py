"""Hierarchical view → temporal → skeleton-joint attention fusion.

Extends the 9.32 mm anchor
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` by replacing
the flat spatio-temporal (time + view) transformer with a three-stage
hierarchy:

1. **Hierarchical view attention** – coarse camera-group attention followed by
   cross-group exchange, so spatially related views reason as a group before all
   views are mixed.
2. **Temporal attention** – motion smoothing over time.
3. **Skeleton-graph joint attention** – anatomy-aware message passing across
   bones/symmetry/cross-view edges.

Everything else (principal-point / focal correction, weight head, DLT
triangulation, residual MLP) is kept unchanged so this is a drop-in paper
ablation of the attention backbone.
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


class _HierarchicalViewTemporalJointBlock(nn.Module):
    """Hierarchical view → temporal → joint-graph attention block.

    Parameters
    ----------
    d:
        Token dimension.
    n_views:
        Number of camera views.
    n_view_groups:
        Number of camera groups in the hierarchy (default 2).
    n_heads:
        Attention heads for the transformer layers.
    n_view_layers:
        Number of within-group view attention layers.
    n_temporal_layers:
        Number of temporal attention layers.
    n_joint_graph_layers:
        Number of skeleton-graph message-passing layers.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        n_view_groups: int,
        n_heads: int,
        n_view_layers: int,
        n_temporal_layers: int,
        n_joint_graph_layers: int,
    ):
        super().__init__()
        if n_view_groups < 1 or n_view_groups > n_views:
            raise ValueError("n_view_groups must be between 1 and n_views")
        self.d = d
        self.n_views = n_views
        self.n_view_groups = n_view_groups

        # Contiguous split of views into groups.
        base = n_views // n_view_groups
        rem = n_views % n_view_groups
        self.group_sizes = [base + (1 if i < rem else 0) for i in range(n_view_groups)]

        # Within-group view attention.
        self.view_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_view_layers)
            ]
        )

        # Cross-group token exchange.
        self.cross_group_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, batch_first=True
        )
        self.cross_group_norm = nn.LayerNorm(d)
        self.cross_group_ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d),
        )
        self.cross_group_ffn_norm = nn.LayerNorm(d)

        # Temporal attention.
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_temporal_layers)
            ]
        )

        # Skeleton-graph joint attention (no view grouping).
        self.joint_graph_layers = nn.ModuleList(
            [
                GraphJointRelation(d=d, n_views=n_views, num_layers=1)
                for _ in range(n_joint_graph_layers)
            ]
        )

    def forward(
        self,
        feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the three-stage hierarchy.

        Args
        ----
        feat:
            ``(B, T, V, J, d)`` spatio-temporal feature tokens.
        edge_index, edge_type:
            Static (view, joint) graph edges for ``GraphJointRelation``.

        Returns
        -------
        ``(B, T, V, J, d)`` refined tokens.
        """
        B, T, V, J, D = feat.shape

        for view_layer in self.view_layers:
            # Within-group view self-attention.
            updated = torch.empty_like(feat)
            start = 0
            for size in self.group_sizes:
                grp = feat[:, :, start : start + size, :, :]  # (B, T, size, J, D)
                # (B, T, size, J, D) -> (B*T*J, size, D)
                grp = grp.permute(0, 1, 3, 2, 4).reshape(B * T * J, size, D)
                grp = view_layer(grp)
                # back to (B, T, size, J, D)
                grp = grp.view(B, T, J, size, D).permute(0, 1, 3, 2, 4)
                updated[:, :, start : start + size, :, :] = grp
                start += size

            # Cross-group attention on pooled per-group tokens.
            group_tokens = []
            start = 0
            for size in self.group_sizes:
                tok = updated[:, :, start : start + size, :, :].mean(dim=2)  # (B, T, J, D)
                group_tokens.append(tok)
                start += size
            group_tokens = torch.stack(group_tokens, dim=2)  # (B, T, G, J, D)
            G = group_tokens.shape[2]

            # Reshape for attention over groups: (B*T*J, G, D)
            gtok = group_tokens.permute(0, 1, 3, 2, 4).reshape(B * T * J, G, D)
            attn_out, _ = self.cross_group_attn(gtok, gtok, gtok)
            gtok = self.cross_group_norm(gtok + attn_out)
            gtok = self.cross_group_ffn_norm(gtok + self.cross_group_ffn(gtok))
            # (B, T, G, J, D)
            gtok = gtok.view(B, T, J, G, D).permute(0, 1, 3, 2, 4)

            # Broadcast the updated group token back to each view in the group.
            start = 0
            for g, size in enumerate(self.group_sizes):
                updated[:, :, start : start + size, :, :] = (
                    updated[:, :, start : start + size, :, :] + gtok[:, :, g : g + 1, :, :]
                )
                start += size

            feat = updated

        # Temporal attention over time for every (view, joint) token.
        for temporal_layer in self.temporal_layers:
            # (B, T, V, J, D) -> (B*V*J, T, D)
            ft = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, D)
            ft = temporal_layer(ft)
            feat = ft.view(B, V, J, T, D).permute(0, 3, 1, 2, 4)

        # Skeleton-graph joint attention over the (view, joint) graph.
        for graph_layer in self.joint_graph_layers:
            fg = feat.reshape(B * T, V, J, D)
            fg = graph_layer(fg, edge_index, edge_type)
            feat = fg.view(B, T, V, J, D)

        return feat


class RayAttentionFusionModelHierarchicalViewTemporalJointResidualPrincipalPoint(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Anchor model with hierarchical view → temporal → skeleton-joint attention.

    Parameters
    ----------
    n_view_groups:
        Number of camera groups in the hierarchical view stage (default 2).
    n_view_layers:
        Number of within-group view transformer layers (default 2).
    n_temporal_layers:
        Number of temporal transformer layers (default 2).
    n_joint_graph_layers:
        Number of skeleton-graph message-passing layers (default 1).
    use_skeleton_graph:
        If ``False``, skip the joint-graph stage (keeps the hierarchy).
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
        return_raw: bool = False,
        n_view_groups: int = 2,
        n_view_layers: int = 2,
        n_temporal_layers: int = 2,
        n_joint_graph_layers: int = 1,
        use_skeleton_graph: bool = True,
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
            return_raw=return_raw,
        )
        self.n_view_groups = n_view_groups
        self.use_skeleton_graph = use_skeleton_graph

        self.hierarchical_block = _HierarchicalViewTemporalJointBlock(
            d=d,
            n_views=n_views,
            n_view_groups=n_view_groups,
            n_heads=n_heads,
            n_view_layers=n_view_layers,
            n_temporal_layers=n_temporal_layers,
            n_joint_graph_layers=n_joint_graph_layers if use_skeleton_graph else 0,
        )

        # Build static (view, joint) graph.
        if j == 17:
            from .graph_joint_relation import (
                H36M_17_PARENTS,
                H36M_17_SYMMETRY_PAIRS,
            )
            parents = H36M_17_PARENTS
            symmetry = H36M_17_SYMMETRY_PAIRS
        elif j == 28:
            parents = MPI_INF_3DHP_28_PARENTS
            symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
        else:
            raise NotImplementedError(
                f"Skeleton graph for J={j} is not implemented. Use J=17 or J=28."
            )
        edge_index, edge_type = build_edge_index(
            parents, symmetry, n_views=n_views, j=j, add_self_loops=True
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device


        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal positional embeddings.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # ---- New: hierarchical view → temporal → skeleton-joint attention ----
        feat = self.hierarchical_block(feat, self.edge_index, self.edge_type)
        feat = feat.reshape(B * T, V, J, self.d)
        # ----------------------------------------------------------------------

        # Optional visibility-aware weighting (base returns 1).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        if self.return_visibility:
            visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            if self.return_visibility:
                visibility = visibility.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(pred_3d_raw.view(B, T, J, 3))
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        if self.return_raw:
            return pred_3d, weights, pred_3d_raw.view(B, T, J, 3)
        return pred_3d, weights
