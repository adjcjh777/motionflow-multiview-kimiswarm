"""Factorised cross-view/temporal ray-aware fusion with residual refinement and
principal-point / focal-length correction.

This combines the factorised attention backbone from
``RayAttentionFusionModelTemporalCrossviewFactorizedResidual`` with the learned
intrinsic-correction layer used by the temporal-only PP model.  It is the
next-step architecture in the ICRA/CVPR 2027 roadmap: more complex attention
(factorised T x V) while preserving the calibration-robust correction head.
"""

import torch
import torch.nn as nn

from .principal_point_correction import PrincipalPointCorrection
from .ray_attention_temporal_crossview_factorized_residual_model import (
    RayAttentionFusionModelTemporalCrossviewFactorizedResidual,
)


class RayAttentionFusionModelTemporalCrossviewFactorizedResidualPrincipalPoint(RayAttentionFusionModelTemporalCrossviewFactorizedResidual):
    """Factorised cross-view/temporal residual model with learned principal-point
    and optional focal-length correction."""

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_view_layers: int = 2,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_view_layers=n_view_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
        )
        self.return_pp_delta = return_pp_delta
        self.correct_focal = focal_max_scale > 0.0
        self.principal_point_correction = PrincipalPointCorrection(
            d=d,
            hidden=principal_point_hidden,
            max_offset=principal_point_max_offset,
            max_focal_scale=focal_max_scale,
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

        # Intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame encoder (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Add positional embeddings.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # Factorised attention: alternating view-level and temporal-level.
        for view_layer, temporal_layer in zip(self.view_layers, self.temporal_layers):
            feat_view = feat.permute(0, 3, 1, 2, 4).reshape(B * T * J, V, self.d)
            feat_view = view_layer(feat_view)
            feat = feat_view.view(B, J, T, V, self.d).permute(0, 3, 2, 1, 4)

            feat_temp = feat.permute(0, 3, 1, 2, 4).reshape(B * V * J, T, self.d)
            feat_temp = temporal_layer(feat_temp)
            feat = feat_temp.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4)

        # Weight head and DLT triangulation.
        feat = feat.reshape(B * T, V, J, self.d)
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)
        weights = weights * confidences
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.return_pp_delta:
            if self.correct_focal:
                return pred_3d, weights, pp_delta, focal_scale
            return pred_3d, weights, pp_delta
        return pred_3d, weights
