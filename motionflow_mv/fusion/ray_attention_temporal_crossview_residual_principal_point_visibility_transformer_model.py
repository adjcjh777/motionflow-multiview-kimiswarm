"""Anchor model with a cross-view visibility transformer.

Subclasses the iter14 best anchor
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and replaces
the simple MLP visibility head with a small transformer that reasons across
views and joints.  The visibility logits multiply the DLT triangulation weights,
explicitly down-weighting occluded or corrupted views while preserving the anchor
geometry (ray-aware features, principal-point correction, spatio-temporal
attention, residual refinement).
"""

import torch
import torch.nn as nn

from .cross_view_visibility_transformer import CrossViewVisibilityTransformer
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibilityTransformer(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + principal-point correction with a
    geometry-aware visibility transformer.

    Parameters
    ----------
    visibility_n_layers:
        Number of transformer layers in the visibility head (default 2).
    visibility_n_heads:
        Number of attention heads in the visibility head (default 4).
    visibility_dropout:
        Dropout probability in the visibility transformer (default 0.1).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining args.
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
        visibility_n_layers: int = 2,
        visibility_n_heads: int = 4,
        visibility_dropout: float = 0.1,
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
            return_visibility=True,
            return_raw=return_raw,
        )
        self.visibility_n_layers = visibility_n_layers
        self.visibility_n_heads = visibility_n_heads
        self.visibility_head = CrossViewVisibilityTransformer(
            d=d,
            n_heads=visibility_n_heads,
            n_layers=visibility_n_layers,
            n_views=n_views,
            j=j,
            dropout=visibility_dropout,
        )

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

        # Prepare per-sample camera tensors and flatten time into batch for the
        # per-frame encoder.
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

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # (B, J, T, V, d) -> (B*J, T*V, d)
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Cross-view visibility transformer.
        visibility_logits = self.visibility_head(feat, confidences)  # (B*T, V, J)
        visibility = torch.sigmoid(visibility_logits)

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
        visibility = visibility.view(B, T, V, J)
        visibility_logits = visibility_logits.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            visibility = visibility.squeeze(1)
            visibility_logits = visibility_logits.squeeze(1)

        raw_3d = pred_3d_raw.view(B, T, J, 3)
        if squeeze_output:
            raw_3d = raw_3d.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.return_visibility:
                out.append(visibility)
                out.append(visibility_logits)
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(raw_3d)
            return tuple(out)

        if self.return_visibility:
            return pred_3d, weights, visibility, visibility_logits
        if self.return_raw:
            return pred_3d, weights, raw_3d
        return pred_3d, weights
