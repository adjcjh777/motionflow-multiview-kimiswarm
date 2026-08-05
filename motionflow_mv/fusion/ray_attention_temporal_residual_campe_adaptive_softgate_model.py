"""Temporal ray-aware fusion with camera PE, residual refinement and a
continuous (soft) adaptive view gate.

The hard Gumbel-top-k selector tends to underperform; this variant replaces it
with a learnable sigmoid gate that softly down-weights unreliable views while
keeping at least ``min_views`` per joint.  A small regulariser encourages the
average number of selected views to be close to ``target_k``.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_campe_model import RayAttentionFusionModelTemporalResidualCamPE
from .ray_attention_model import _triangulate_weighted_dlt


class RayAttentionFusionModelTemporalResidualCamPEAdaptiveSoftGate(RayAttentionFusionModelTemporalResidualCamPE):
    """CamPE residual model with a continuous adaptive view gate.

    Parameters
    ----------
    target_k:
        Desired average number of active views per joint (used only by the
        regulariser, not a hard constraint).
    min_views:
        Minimum number of views that must survive the gate; enforced via a
        soft ReLU penalty.
    lambda_gate:
        Weight of the gate regulariser in the loss.  Set to ``0.0`` to disable.
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
        target_k: int = 4,
        min_views: int = 2,
        lambda_gate: float = 0.01,
        **kwargs,
    ):
        # Bypass the dense joint attention from the base model; we keep the
        # same view-level + temporal encoder as CamPE.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            n_bands=n_bands,
        )
        self.target_k = target_k
        self.min_views = min_views
        self.lambda_gate = lambda_gate

        self.score_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
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

        # Soft adaptive gate.
        scores = self.score_mlp(feat).squeeze(-1)  # (N, V, J)
        gate = torch.sigmoid(scores)  # (N, V, J)
        weights = weights * gate

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
        gate = gate.view(B, T, V, J)

        # Regulariser: encourage target_k active views and at least min_views.
        avg_active = gate.mean(dim=1)  # (B, T, J) actually after reshape gate is (B,T,V,J); mean over V -> (B,T,J)
        gate_reshaped = gate.view(B * T, V, J)
        avg_active = gate_reshaped.mean(dim=1)  # (B*T, J)
        target = self.target_k / float(V)
        reg_target = ((avg_active - target) ** 2).mean()
        sum_active = gate_reshaped.sum(dim=1)  # (B*T, J)
        reg_min = torch.relu(float(self.min_views) - sum_active).mean()
        reg = reg_target + reg_min

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            gate = gate.squeeze(1)

        return pred_3d, weights, gate, reg
