"""Ray-aware attention fusion v2: view attention + cross-joint attention.

Builds on `RayAttentionFusionModel` by adding a second transformer layer over
joints after the per-view attention. This lets the model propagate anatomical
constraints across joints before triangulation, which is especially useful
when some joints are occluded or corrupted.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_model import _compute_rays
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelV2(nn.Module):
    """Ray-aware attention fusion with view-level + joint-level attention.

    Input / output semantics are identical to `RayAttentionFusionModel`.
    """

    def __init__(self, j: int = 17, d: int = 64, n_views: int = 4, n_heads: int = 4, n_joint_layers: int = 1):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        # View-level self-attention: each joint attends over views.
        self.view_attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, batch_first=True)
        self.view_norm1 = nn.LayerNorm(d)
        self.view_ffn = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.view_norm2 = nn.LayerNorm(d)

        # Joint-level self-attention: each view attends over joints (skeleton-aware).
        self.joint_attn = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True)
             for _ in range(n_joint_layers)]
        )

        # Fuse multi-view per-joint representation.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        self.weight_head = nn.Linear(d, 1)

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
    ):
        B, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B, -1, -1, -1)
            R = R.unsqueeze(0).expand(B, -1, -1, -1)
            t = t.unsqueeze(0).expand(B, -1, -1)

        points_2d = x[..., :2]
        confidences = x[..., 2]

        # Ray features (same as v1).
        rays = _compute_rays(points_2d, K, R, t)  # (B, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (B, V, 3)
        centers_expanded = centers[:, :, None, :].expand(B, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (B, V, J, 6)

        obs_emb = self.obs_embed(x)  # (B, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (B, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (B, V, J, d)

        # View-level attention: per joint, views as sequence.
        feat_v = feat.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
        attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
        feat_v = self.view_norm1(feat_v + attn_out)
        feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))  # (B*J, V, d)
        feat_v = feat_v.view(B, J, V, self.d)  # (B, J, V, d)

        # Joint-level attention: per view, joints as sequence.
        feat_j = feat_v.permute(0, 2, 1, 3).reshape(B * V, J, self.d)
        for layer in self.joint_attn:
            feat_j = layer(feat_j)
        feat_j = feat_j.view(B, V, J, self.d).permute(0, 2, 1, 3)  # (B, J, V, d)

        # Fuse views per joint.
        feat_jv = feat_j.reshape(B, J, V * self.d)
        feat_fused = self.fusion_mlp(feat_jv)  # (B, J, d)

        # Predict per-view weights per joint.
        w_logits = self.weight_head(feat_j).squeeze(-1)  # (B, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B, V, J)
        weights = weights * confidences

        # Differentiable weighted DLT.
        Rt = torch.cat([R, t[:, :, :, None]], dim=-1)  # (B, V, 3, 4)
        P = K @ Rt
        pred_3d = self._triangulate_weighted_dlt(points_2d, weights, P)
        return pred_3d, weights

    def _triangulate_weighted_dlt(self, points_2d, weights, proj_matrices):
        from .ray_attention_model import _triangulate_weighted_dlt
        return _triangulate_weighted_dlt(points_2d, weights, proj_matrices)
