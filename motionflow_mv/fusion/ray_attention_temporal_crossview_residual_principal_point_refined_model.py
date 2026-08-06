"""Temporal + cross-view ray-aware attention with refined principal-point correction.

The model first predicts a coarse principal-point correction from raw 2D
observations, extracts spatio-temporal features, and then predicts a residual
principal-point correction from the pooled cross-view features. Triangulation
with the refined intrinsics yields the final 3D pose.
"""

import torch
import torch.nn as nn

from .principal_point_correction import PrincipalPointCorrection
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointRefined(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual model with two-stage principal-point correction.

    Stage 1: coarse correction from raw observations (before feature extraction).
    Stage 2: residual correction from pooled spatio-temporal features (after
    cross-view attention), followed by re-triangulation.
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
        refined_principal_point_hidden: int = 64,
        refined_principal_point_max_offset: float = 10.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
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
        )
        self.refined_principal_point_correction = PrincipalPointCorrection(
            d=d,
            hidden=refined_principal_point_hidden,
            max_offset=refined_principal_point_max_offset,
            max_focal_scale=0.0,  # keep it simple: residual PP only
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

        # Prepare per-sample camera tensors and flatten time into batch.
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

        # Stage 1: coarse principal-point / intrinsic correction.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame features with coarse corrected intrinsics.
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction with coarse corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences
        weights = weights.clamp(min=1e-4)

        # Stage 2: residual PP correction from per-joint spatio-temporal features.
        K_refined, pp_delta_refined = self.refined_principal_point_correction(
            K=K_corrected,
            feat=feat,  # (B*T, V, J, d)
            weights=None,
        )

        # Triangulate with refined intrinsics.
        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)
        P_refined = K_refined @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P_refined)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.return_pp_delta:
            outputs = [pred_3d, weights, pp_delta, pp_delta_refined]
            if self.correct_focal:
                outputs.append(focal_scale)
            return tuple(outputs)
        return pred_3d, weights
