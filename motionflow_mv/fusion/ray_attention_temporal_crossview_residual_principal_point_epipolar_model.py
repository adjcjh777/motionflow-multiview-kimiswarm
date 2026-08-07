"""Anchor PP model with an epipolar-line distance bias on the weight head.

This injects calibrated geometry into the cross-view fusion by down-weighting
view pairs whose keypoints are inconsistent with the epipolar constraint.
"""

import torch
import torch.nn as nn

from .epipolar_attention_bias import (
    compute_epipolar_distance,
    epipolar_bias_from_distance,
)
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """PP anchor with epipolar bias on the per-view weight head."""

    def __init__(self, *args, epipolar_temperature: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epipolar_temperature = epipolar_temperature
        # Optional scalar gate to blend epipolar bias with unbiased logits.
        self.epipolar_gate = nn.Parameter(torch.zeros(1))

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

        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        feat = self._extract_frame_features(x_flat, K_corrected, R, t)
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        visibility = self._visibility_multiplier(feat, confidences)

        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)

        # Epipolar bias on weight logits.
        epi_dist = compute_epipolar_distance(K_corrected, R, t, points_2d)
        epi_bias = epipolar_bias_from_distance(epi_dist, temperature=self.epipolar_temperature)
        epi_bias = epi_bias.permute(0, 2, 1)  # (B*T, J, V)
        gate = torch.sigmoid(self.epipolar_gate)
        w_logits = w_logits + gate * epi_bias

        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        feat_pooled = feat.mean(dim=1)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.residual_mlp(residual_input)
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

        raw_3d = pred_3d_raw.view(B, T, J, 3)
        if squeeze_output:
            raw_3d = raw_3d.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(raw_3d)
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        if self.return_raw:
            return pred_3d, weights, raw_3d
        return pred_3d, weights
