"""Multi-person extension of the best single-person anchor.

Adds a cross-person association graph between the spatio-temporal transformer
and the triangulation head so that multiple people can be fused jointly across
views while resolving occlusions and identity ambiguities.
"""

import torch

from .multiperson_association_graph import MultiPersonAssociationGraph
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointMultiPersonAssoc(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP model with multi-person association graph.

    Accepts either the original single-person tensor
    ``(B, T, V, J, 3)`` or a multi-person tensor ``(B, T, V, P, J, 3)``.
    When ``P == 1`` the graph is disabled and the module degenerates to the
    parent anchor.

    Parameters
    ----------
    n_persons:
        Number of people ``P`` in the multi-person clip (default ``1``).
    assoc_num_layers:
        Number of message-passing layers in the association graph (default ``2``).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_persons: int = 1,
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
        assoc_num_layers: int = 2,
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
        self.n_persons = n_persons
        self.assoc_num_layers = assoc_num_layers

        if n_persons > 1:
            self.assoc_graph = MultiPersonAssociationGraph(
                d=d,
                n_views=n_views,
                n_persons=n_persons,
                num_layers=assoc_num_layers,
            )
        else:
            self.assoc_graph = None

    def _skeleton_edges(self, j: int):
        """Return (parents, symmetry_pairs) for the given joint count."""
        if j == 17:
            from .graph_joint_relation import (
                H36M_17_PARENTS,
                H36M_17_SYMMETRY_PAIRS,
            )
            return H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS
        elif j == 28:
            from .graph_joint_relation import (
                MPI_INF_3DHP_28_PARENTS,
                MPI_INF_3DHP_28_SYMMETRY_PAIRS,
            )
            return MPI_INF_3DHP_28_PARENTS, MPI_INF_3DHP_28_SYMMETRY_PAIRS
        else:
            raise NotImplementedError(
                f"Multi-person graph skeleton for J={j} is not implemented."
            )

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        # Single-person input: fall back to the parent model unchanged.
        if x.dim() in (4, 5):
            return super().forward(x, cameras=cameras, K=K, R=R, t=t)

        if x.dim() != 6:
            raise ValueError(
                "Input must be (B, T, V, J, 3) single-person or (B, T, V, P, J, 3) multi-person."
            )

        B, T, V, P, J, _ = x.shape
        device = x.device


        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

        # Expand camera matrices over the batch/time dimensions.
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

        # Reshape multi-person input to a single batch dimension over people.
        # x: (B, T, V, P, J, 3) -> (B*P, T, V, J, 3)
        x_perm = x.permute(0, 3, 1, 2, 4, 5)  # (B, P, T, V, J, 3)
        x_flat = x_perm.reshape(B * P, T, V, J, 3)
        x_flat = x_flat.reshape(B * P * T, V, J, 3)

        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction (per-person).
        correction_outputs = self.principal_point_correction(
            K=K.repeat_interleave(P, dim=0),
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame features.
        feat = self._extract_frame_features(
            x_flat,
            K_corrected,
            R.repeat_interleave(P, dim=0),
            t.repeat_interleave(P, dim=0),
        )  # (B*P*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B * P, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * P * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B * P, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(
            B * P * T, V, J, self.d
        )

        # Multi-person association graph over (view, person, joint) nodes.
        if self.assoc_graph is not None and P > 1:
            # (B*P*T, V, J, d) -> (B*T, V, P, J, d)
            feat_assoc = feat.view(B, P, T, V, J, self.d)
            feat_assoc = feat_assoc.permute(0, 3, 2, 1, 4, 5).reshape(
                B * T, V, P, J, self.d
            )
            parents, sym_pairs = self._skeleton_edges(J)
            feat_assoc = self.assoc_graph(feat_assoc, parents, sym_pairs)
            feat = feat_assoc.view(B, T, V, P, J, self.d)
            feat = feat.permute(0, 4, 1, 2, 5, 3).reshape(
                B * P * T, V, J, self.d
            )

        # Optional visibility-aware weighting.
        visibility = self._visibility_multiplier(feat, confidences)

        # Per-frame weight prediction and triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*P*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*P*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*P*T, V, J)
        weights = weights * confidences * visibility
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt

        Rt = torch.cat(
            [R.repeat_interleave(P, dim=0), t.repeat_interleave(P, dim=0)[..., None]],
            dim=-1,
        )  # (B*P*T, V, 3, 4)
        P_mat = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P_mat)  # (B*P*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*P*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*P*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*P*T, J, 3)
        pred_3d = pred_3d_raw + delta

        # Reshape back to multi-person output.
        pred_3d = pred_3d.view(B, P, T, J, 3).permute(0, 2, 1, 3, 4)  # (B, T, P, J, 3)
        weights = weights.view(B, P, T, V, J).permute(0, 2, 3, 1, 4)  # (B, T, V, P, J)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        return pred_3d, weights
