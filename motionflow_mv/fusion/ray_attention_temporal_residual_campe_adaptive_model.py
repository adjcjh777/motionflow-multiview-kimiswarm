"""Temporal ray-aware fusion with camera PE, residual refinement and adaptive view selection."""

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_campe_model import RayAttentionFusionModelTemporalResidualCamPE
from .adaptive_view_selector import AdaptiveViewSelector
from .ray_attention_model import _triangulate_weighted_dlt


class RayAttentionFusionModelTemporalResidualCamPEAdaptive(RayAttentionFusionModelTemporalResidualCamPE):
    """Temporal ray-aware fusion with camera PE, residual refinement and adaptive view selection.

    The selector predicts a per-view, per-joint binary mask from the encoder
    tokens. During training a Gumbel-softmax mask is multiplied into the DLT
    weights; during inference a hard top-k mask is used.

    Parameters
    ----------
    k:
        Number of views to select at inference time.
    selector_tau:
        Gumbel-softmax temperature during training.
    geo_features:
        Whether to augment selector tokens with ray-geometry features.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        n_bands: int = 4,
        k: int = 4,
        selector_tau: float = 0.5,
        geo_features: bool = True,
        **kwargs,
    ):
        super().__init__(
            j=j, d=d, n_views=n_views, n_heads=n_heads,
            n_joint_layers=n_joint_layers, n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len, residual_hidden=residual_hidden,
            n_bands=n_bands,
        )
        self.k = k
        self.selector = AdaptiveViewSelector(
            d=d, n_views=n_views, k=k, tau=selector_tau, geo_features=geo_features, hard_inference=True,
        )

    def forward(self, x, cameras=None, K=None, R=None, t=None, n_iter: int = 1):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_model import _cameras_to_tensors
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

        feat = self._extract_frame_features(x_flat, K, R, t)

        # Temporal attention.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Weight head.
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)
        weights = weights * confidences

        # Adaptive view selection.
        _, select_mask, _ = self.selector(feat, points_2d=points_2d, K=K, R=R, t=t)
        weights = weights * select_mask

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_raw
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights
