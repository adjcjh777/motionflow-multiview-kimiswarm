"""Anchor PP model with an epipolar-geometry bias inside the ST transformer (v2).

Where v1 (``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolar``)
applies epipolar consistency only to the final per-view weight head, this v2 model
injects the same epipolar consistency as a *relative-position bias* into each
layer of the spatio-temporal transformer.  The transformer therefore fuses
multi-view tokens in a way that is explicitly conditioned on calibrated geometry,
improving robustness to noisy views while preserving the triangulation pipeline.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from .epipolar_transformer_bias import (
    compute_per_frame_epipolar_bias,
    EpipolarBiasedTransformerEncoderLayer,
    build_temporal_bias_from_frames,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarBiasV2(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Cross-view temporal residual model with epipolar-biased ST transformer.

    Parameters
    ----------
    epipolar_temperature:
        Divisor applied to the symmetric epipolar distance before the attention
        bias is computed (default 10.0; smaller = sharper bias).
    gate_init:
        Initial logit for the learned gate that blends the epipolar bias with
        the standard positional embedding pathway.  A small positive value keeps
        the bias active from the start.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        *args,
        epipolar_temperature: float = 10.0,
        gate_init: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epipolar_temperature = epipolar_temperature
        # Learned scalar gate that interpolates the epipolar bias; sigmoid keeps
        # the blend in (0, 1).  A positive init biases the network toward using
        # the geometry term early in training.
        self.epipolar_gate = nn.Parameter(torch.full((1,), gate_init))

        # Replace the standard ST transformer layers with geometry-aware layers.
        # The architecture exactly mirrors the parent model.
        self.st_transformer = nn.ModuleList([
            EpipolarBiasedTransformerEncoderLayer(
                d_model=self.d,
                nhead=self.n_heads,
                dim_feedforward=self.d * 2,
                dropout=0.1,  # matches parent nn.TransformerEncoderLayer default
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(len(self.st_transformer))
        ])

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

        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Add standard positional embeddings (kept from parent model).
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # Build epipolar attention bias from the corrected intrinsics.  The bias
        # is block-diagonal in time: same-frame view pairs are encouraged when
        # they are epipolarly consistent; cross-frame pairs are left unbiased.
        with torch.no_grad():
            per_frame_bias = compute_per_frame_epipolar_bias(
                K_corrected, R, t, points_2d, temperature=self.epipolar_temperature
            )  # (B*T, V, V)
            per_frame_bias = per_frame_bias.view(B, T, V, V)
        attn_bias = build_temporal_bias_from_frames(per_frame_bias, n_heads=self.n_heads, n_joints=J)

        # (B, J, T, V, d) -> (B*J, T*V, d)
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        gate = torch.sigmoid(self.epipolar_gate)
        for layer in self.st_transformer:
            feat = layer(feat, epipolar_bias=gate * attn_bias)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Remainder is identical to the parent anchor model.
        visibility = self._visibility_multiplier(feat, confidences)

        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

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
