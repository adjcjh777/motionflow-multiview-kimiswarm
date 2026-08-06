"""Temporal ray-aware attention fusion with residual refinement and uncertainty.

Extends ``RayAttentionFusionModelTemporalResidual`` by making the weight head
predict two values per view/joint:

* a per-view per-joint weight used for DLT triangulation, and
* a per-view per-joint uncertainty used to build an uncertainty-aware pooled
  feature for the residual refinement head.

The residual head therefore reasons about both the raw triangulated 3D pose
and a per-joint uncertainty summary, allowing it to down-weight or up-weight
joints based on the model's own confidence estimate.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    X: (B, T, J, 3) refined world-coordinate 3D joints, or (B, J, 3) for 4D input
    weights: (B, T, V, J) predicted per-view per-joint weights, or (B, V, J)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .ray_attention_temporal_model import _cameras_to_tensors
from .ray_attention_model import _triangulate_weighted_dlt
from ..calibration.camera import Camera


class RayAttentionFusionModelTemporalResidualUncertainty(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-attention fusion with uncertainty-aware residual refinement.

    The weight head now outputs two quantities per view/joint:

    * ``weights`` (sigmoid-activated) are multiplied by the input confidence
      and fed to weighted DLT triangulation.
    * ``uncertainty`` (softplus-activated, positive) is used to build an
      inverse-uncertainty-weighted per-joint feature for the residual head.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_temporal_layers, max_temporal_len,
    residual_hidden:
        See ``RayAttentionFusionModelTemporalResidual``.
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
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
        )

        # Weight head now predicts (weight, uncertainty) per view/joint.
        self.weight_head = nn.Linear(self.d, 2)

        # Residual head: d feature + 3D raw pose + 1 per-joint uncertainty summary.
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.d + 3 + 1, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
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

        # Per-frame v3 features.
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight/uncertainty prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        weight_out = self.weight_head(feat_for_weight)  # (B*T, J, V, 2)
        w_logits = weight_out[..., 0]  # (B*T, J, V)
        u_logits = weight_out[..., 1]  # (B*T, J, V)

        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)

        # Positive per-view per-joint uncertainty.
        uncertainty = F.softplus(u_logits).permute(0, 2, 1) + 1e-6  # (B*T, V, J)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Uncertainty-aware pooling of per-view temporal features.
        # Weight each view by inverse uncertainty, then concatenate a summary.
        inv_u = 1.0 / uncertainty  # (B*T, V, J)
        u_weights = inv_u / inv_u.sum(dim=1, keepdim=True)  # (B*T, V, J)
        feat_pooled = (feat * u_weights[..., None]).sum(dim=1)  # (B*T, J, d)

        # Per-joint log-uncertainty summary (mean across views).
        log_unc_summary = torch.log(uncertainty).mean(dim=1)  # (B*T, J)

        # Residual refinement head.
        residual_input = torch.cat(
            [feat_pooled, pred_3d_raw, log_unc_summary.unsqueeze(-1)], dim=-1
        )  # (B*T, J, d + 3 + 1)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights


# Backward-compatible alias (renamed to avoid collision with the diagnostic v3 class).
RayAttentionFusionModelTemporalResidualV3 = RayAttentionFusionModelTemporalResidualUncertainty


if __name__ == "__main__":
    # Quick shape/gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    from .ray_attention_temporal_residual_model import _make_cameras

    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualUncertainty(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("temporal residual v3 uncertainty model sanity check passed")
