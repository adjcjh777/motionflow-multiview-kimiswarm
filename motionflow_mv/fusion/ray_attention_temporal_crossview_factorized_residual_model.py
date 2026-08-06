"""Temporal ray-aware fusion with factorised cross-view + temporal attention.

The original cross-view model flattens the (time, view) grid and attends over it
in one large transformer.  This variant factorises the operation into alternating
view-level attention (for each joint and timestep) and temporal-level attention
(for each view and joint).  This preserves axial structure and is usually more
parameter-efficient for the same capacity.

Input / output semantics are identical to ``RayAttentionFusionModelTemporalCrossviewResidual``.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_model import _compute_rays, _triangulate_weighted_dlt


def _cameras_to_tensors(cameras, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelTemporalCrossviewFactorizedResidual(nn.Module):
    """Factorised cross-view/temporal ray-aware fusion with residual refinement."""

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
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        self.camera_embed_mlp = nn.Sequential(
            nn.Linear(9 + 9 + 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        self.view_attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, batch_first=True)
        self.view_norm1 = nn.LayerNorm(d)
        self.view_ffn = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.view_norm2 = nn.LayerNorm(d)

        self.joint_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_joint_layers)
            ]
        )

        self.time_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.view_pos_embed = nn.Parameter(torch.randn(n_views, d) * 0.02)

        # Factorised blocks: alternating view-level and temporal-level attention.
        self.view_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_view_layers)
            ]
        )
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_temporal_layers)
            ]
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.weight_head = nn.Linear(d, 1)

        self.residual_hidden = residual_hidden
        self.residual_mlp = nn.Sequential(
            nn.Linear(d + 3, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 3),
        )

    def _extract_frame_features(self, x, K, R, t):
        """Run the per-frame encoder. Input (N, V, J, 3); output (N, V, J, d)."""
        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        rays = _compute_rays(points_2d, K, R, t)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)

        obs_emb = self.obs_embed(x)
        ray_emb = self.ray_embed(ray_input)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)

        camera_feat = torch.cat([K.view(N, V, -1), R.view(N, V, -1), t.view(N, V, -1)], dim=-1)
        camera_emb = self.camera_embed_mlp(camera_feat)
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

        return feat_j

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

        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Add positional embeddings.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # Factorised attention: alternating view-level and temporal-level.
        for view_layer, temporal_layer in zip(self.view_layers, self.temporal_layers):
            # View attention: (B*T*J, V, d)
            feat_view = feat.permute(0, 3, 1, 2, 4).reshape(B * T * J, V, self.d)
            feat_view = view_layer(feat_view)
            feat = feat_view.view(B, J, T, V, self.d).permute(0, 3, 2, 1, 4)  # (B, V, T, J, d)

            # Temporal attention: (B*V*J, T, d)
            feat_temp = feat.permute(0, 3, 1, 2, 4).reshape(B * V * J, T, self.d)
            feat_temp = temporal_layer(feat_temp)
            feat = feat_temp.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4)  # (B, T, V, J, d)

        # Weight head and DLT triangulation.
        feat = feat.reshape(B * T, V, J, self.d)
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        feat_pooled = feat.mean(dim=1)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights
