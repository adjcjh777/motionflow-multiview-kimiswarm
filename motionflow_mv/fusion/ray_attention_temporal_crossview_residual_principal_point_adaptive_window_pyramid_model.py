"""Adaptive temporal-window pyramid for calibrated multi-view 3D pose.

Subclasses the 9.32 mm anchor
(``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``) and replaces
the single spatio-temporal transformer block with a stack of adaptive
multi-scale temporal attention layers.  Short windows capture fast motion,
medium windows capture smooth articulation, and a global window provides
long-range temporal context.  A learned per-token scale gate adaptively mixes
the outputs.

The change is purely local to the temporal fusion stage: principal-point
(focal) correction, per-frame view/joint encoder, weight head, and residual
refinement head are inherited unchanged.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class AdaptiveWindowPyramidLayer(nn.Module):
    """Multi-scale temporal attention with learned scale mixing.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of camera views (kept for API compatibility, not a hard
        architectural parameter).
    temporal_scales:
        Tuple of window sizes.  ``0`` is interpreted as "full clip" (global
        temporal attention).  Odd window sizes are recommended so the window is
        symmetric around each frame.
    n_heads:
        Number of attention heads for each scale-specific encoder layer.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        temporal_scales: Tuple[int, ...] = (3, 7, 0),
        n_heads: int = 4,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.temporal_scales = temporal_scales

        # One attention block per temporal scale.  Each block consumes
        # (B*V*J, T, d) tokens and attends within the requested local window.
        self.scale_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in temporal_scales
            ]
        )

        # Learned scale-mixing gate: per (time, view, joint) token, produces a
        # convex weighting over scales from the concatenated scale features.
        self.gate = nn.Sequential(
            nn.Linear(d * len(temporal_scales), d),
            nn.ReLU(),
            nn.Linear(d, len(temporal_scales)),
            nn.Softmax(dim=-1),
        )

    def _local_mask(self, T: int, window: int, device: torch.device) -> torch.Tensor:
        """Build a (T, T) additive attention mask for a local temporal window.

        ``float('-inf')`` marks positions outside the window and ``0.0`` marks
        positions inside.  ``window <= 0`` or ``window >= T`` means global
        attention and returns ``None``.
        """
        if window <= 0 or window >= T:
            return None
        half = window // 2
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            start = max(0, i - half)
            end = min(T, i + half + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale temporal attention and fuse with a learned gate.

        Args
        ----
        feat:
            ``(B, T, V, J, d)`` spatio-temporal features after time/view
            positional embeddings.

        Returns
        -------
        fused:
            ``(B, T, V, J, d)`` fused features.
        """
        B, T, V, J, d = feat.shape

        # Process each scale independently.  Each token sequence is organised
        # as (B*V*J, T, d) so the transformer sees only temporal neighbours.
        scale_outputs = []
        for window, attn in zip(self.temporal_scales, self.scale_attn):
            feat_in = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, d)
            mask = self._local_mask(T, window, feat.device)
            out = attn(feat_in, src_mask=mask)  # (B*V*J, T, d)
            out = out.reshape(B, V, J, T, d).permute(0, 3, 1, 2, 4)  # (B, T, V, J, d)
            scale_outputs.append(out)

        # Stack outputs: (B, T, V, J, S, d)
        stacked = torch.stack(scale_outputs, dim=-2)

        # Gate from concatenated scale features per token.
        gate_input = stacked.reshape(B, T, V, J, -1)  # (B, T, V, J, S*d)
        scale_weights = self.gate(gate_input)  # (B, T, V, J, S)

        fused = (scale_weights.unsqueeze(-1) * stacked).sum(dim=-2)  # (B, T, V, J, d)
        return fused


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Anchor model with an adaptive temporal-window pyramid fusion stage.

    Parameters
    ----------
    temporal_scales:
        Window sizes for the pyramid.  ``0`` means full-clip global attention.
        Default is ``(3, 7, 0)``.
    pyramid_layers:
        Number of stacked pyramid layers (default 1).
    pyramid_n_heads:
        Number of attention heads in each scale-specific block (default 4).
    return_scale_weights:
        If ``True``, also return the last pyramid layer's scale weights for
        diagnostics.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    all remaining parameters.
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
        temporal_scales: Tuple[int, ...] = (3, 7, 0),
        pyramid_layers: int = 1,
        pyramid_n_heads: int = 4,
        return_scale_weights: bool = False,
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
        self.temporal_scales = temporal_scales
        self.pyramid_layers = nn.ModuleList(
            [
                AdaptiveWindowPyramidLayer(
                    d=d,
                    n_views=n_views,
                    temporal_scales=temporal_scales,
                    n_heads=pyramid_n_heads,
                )
                for _ in range(pyramid_layers)
            ]
        )
        self.return_scale_weights = return_scale_weights

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

        # Spatio-temporal (time + view) attention replaced by adaptive pyramid.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        for layer in self.pyramid_layers:
            feat = layer(feat)

        # Reshape back to the per-frame layout the rest of the anchor expects.
        feat = feat.reshape(B * T, V, J, self.d)

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

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_visibility:
                raise NotImplementedError("return_visibility not implemented in pyramid model")
            if self.return_scale_weights:
                out.append(scale_weights)
            if self.return_raw:
                out.append(pred_3d_raw.view(B, T, J, 3))
            return tuple(out)

        if self.return_visibility:
            raise NotImplementedError("return_visibility not implemented in pyramid model")

        if self.return_scale_weights:
            return pred_3d, weights, scale_weights

        return pred_3d, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, V, J = 2, 13, 4, 17
    x = torch.randn(B, T, V, J, 3)
    x[..., 2] = (x[..., 2] + 1).clamp(min=0.0)

    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1).float()
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1).float()
    t = torch.zeros(V, 3).float()

    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointAdaptiveWindowPyramid(
        j=J,
        d=64,
        n_views=V,
        n_st_layers=2,
        residual_hidden=128,
        temporal_scales=(3, 7, 0),
        pyramid_layers=1,
        return_pp_delta=True,
    )

    pred, weights, pp_delta = model(x, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    print("adaptive temporal window pyramid model sanity check passed")
