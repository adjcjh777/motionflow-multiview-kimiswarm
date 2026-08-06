"""Temporal ray-aware attention fusion with uncertainty-weighted DLT.

Extends ``RayAttentionFusionModelTemporal`` by replacing the sigmoid per-view
weight head with a Gaussian uncertainty head.  For each view and joint the model
predicts a log-variance ``log_var``.  The weighted DLT then uses

    weight = confidence * exp(-log_var)

so that views predicted to be uncertain receive lower weight.  An optional
per-view reprojection NLL auxiliary loss is computed to encourage the predicted
uncertainties to match the actual reprojection errors.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_model import _compute_rays, _triangulate_weighted_dlt
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelTemporalUncertainty(nn.Module):
    """Temporal ray-aware fusion with view/joint/temporal attention and
    uncertainty-weighted triangulation.

    Input:
        x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
        cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
        OR
        K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

    Output (tuple):
        pred_3d: (B, T, J, 3) or (B, J, 3) world-coordinate 3D joints
        weights: (B, T, V, J) or (B, V, J) per-view per-joint DLT weights
        log_var: (B, T, V, J) or (B, V, J) predicted log variance per view/joint
        nll_loss: scalar auxiliary reprojection NLL loss (0 if not computed)
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
        uncertainty_loss_weight: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Per-frame embeddings.
        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        # Camera-conditioned embedding.
        self.camera_embed_mlp = nn.Sequential(
            nn.Linear(9 + 9 + 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # View-level self-attention.
        self.view_attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, batch_first=True)
        self.view_norm1 = nn.LayerNorm(d)
        self.view_ffn = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.view_norm2 = nn.LayerNorm(d)

        # Joint-level self-attention.
        self.joint_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_joint_layers)
            ]
        )

        # Temporal position embedding and temporal transformer.
        self.temporal_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.temporal_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_temporal_layers)
            ]
        )

        # Fusion MLP and uncertainty head.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.uncertainty_head = nn.Linear(d, 1)

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the v3 per-frame encoder.  Input shape (N, V, J, 3); output (N, V, J, d)."""
        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (N, V, 3)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (N, V, J, 6)

        obs_emb = self.obs_embed(x)  # (N, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (N, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (N, V, J, d)

        # Camera embedding.
        camera_feat = torch.cat([K.view(N, V, -1), R.view(N, V, -1), t.view(N, V, -1)], dim=-1)  # (N, V, 21)
        camera_emb = self.camera_embed_mlp(camera_feat)  # (N, V, d)
        camera_emb = camera_emb[:, :, None, :].expand(N, V, J, self.d)
        feat = feat + camera_emb

        # View-level attention.
        feat_v = feat.permute(0, 2, 1, 3).reshape(N * J, V, self.d)
        attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
        feat_v = self.view_norm1(feat_v + attn_out)
        feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))
        feat_v = feat_v.view(N, J, V, self.d)

        # Joint-level attention.
        feat_j = feat_v.permute(0, 2, 1, 3).reshape(N * V, J, self.d)
        for layer in self.joint_attn:
            feat_j = layer(feat_j)
        feat_j = feat_j.view(N, V, J, self.d)

        return feat_j  # (N, V, J, d)

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood.

        Args:
            points_2d: (N, V, J, 2)
            pred_3d: (N, J, 3)
            proj_matrices: (N, V, 3, 4)
            log_var: (N, V, J)

        Returns:
            nll: scalar
        """
        N, V, J, _ = points_2d.shape
        # Homogeneous 3D points: (N, J, 4)
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)  # (N, J, 4)
        # Project: (N, V, J, 3)
        p_h = torch.einsum('nvij,nkj->nvki', proj_matrices, Xh)
        # Avoid division by zero / behind-camera points.
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z  # (N, V, J, 2)
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)  # (N, V, J)
        # Gaussian NLL up to constants: 0.5 * (err^2 / var + log_var)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

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

        # Per-frame uncertainty prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        log_var = self.uncertainty_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)  # (B*T, V, J)

        # Use precision as the DLT weight, scaled by observed confidence.
        precision = torch.exp(-log_var)
        weights = precision * confidences  # (B*T, V, J)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Auxiliary reprojection NLL so uncertainties are supervised.
        nll_loss = self._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = self.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        log_var = log_var.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            log_var = log_var.squeeze(1)

        return pred_3d, weights, log_var, nll_loss
