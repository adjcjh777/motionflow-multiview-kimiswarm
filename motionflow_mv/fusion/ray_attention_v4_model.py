"""Ray-aware attention fusion v4: normalized camera-geometry embeddings.

Builds on `RayAttentionFusionModelV3` by replacing the raw flattened
``K/R/t`` camera embedding with a geometrically normalized camera token:

    [normalized focal length, normalized principal point,
     6-D rotation (first two columns of R), scaled translation]

The normalization makes the embedding invariant to absolute scene scale and
camera resolution, which should help the model generalize across different
camera rigs and datasets.

Summary of changes (2026-08-04 swarm turn):
* New `_normalized_camera_embedding()` extracts 13-D camera tokens from
  intrinsics/extrinsics.
* Focal lengths and principal points are normalized by the mean focal length.
* Rotation is represented by the first two columns of R (6-D continuous
  rotation embedding).
* Translation is scaled by the mean camera distance so rigs with different
  metric scales are embedded similarly.
* The rest of the architecture (view attention, joint attention, weighted
  DLT) is unchanged from v3.
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


def _normalized_camera_embedding(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a normalized per-view camera embedding.

    Args:
        K: (B, V, 3, 3) intrinsic matrices.
        R: (B, V, 3, 3) rotation matrices.
        t: (B, V, 3) translation vectors.

    Returns:
        camera_feat: (B, V, 13) concatenation of
            [fx/f, fy/f, cx/f, cy/f, R[:, :2].flat, t / mean_t_norm]
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    f_mean = (fx + fy) * 0.5
    focal = torch.stack([fx / (f_mean + eps), fy / (f_mean + eps)], dim=-1)  # (B, V, 2)
    principal = torch.stack([cx / (f_mean + eps), cy / (f_mean + eps)], dim=-1)  # (B, V, 2)

    # 6-D rotation: first two columns of R (Zhou et al., On the Continuity of
    # Rotation Representations in Neural Networks).
    R_cols = R[..., :2].reshape(*R.shape[:-2], 6)  # (B, V, 6)

    # Scale translation by the average camera distance for scale invariance.
    t_norm = t.norm(dim=-1)  # (B, V)
    mean_t_norm = t_norm.mean(dim=-1, keepdim=True)  # (B, 1)
    t_scaled = t / (mean_t_norm.unsqueeze(-1) + eps)  # (B, V, 3)

    return torch.cat([focal, principal, R_cols, t_scaled], dim=-1)


class RayAttentionFusionModelV4(nn.Module):
    """Ray-aware attention fusion with normalized camera-geometry embeddings."""

    def __init__(self, j: int = 17, d: int = 64, n_views: int = 4, n_heads: int = 4, n_joint_layers: int = 1):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        # Normalized camera embedding: 13-D -> d
        self.camera_embed_mlp = nn.Sequential(
            nn.Linear(13, d),
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
            [nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True)
             for _ in range(n_joint_layers)]
        )

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

        rays = _compute_rays(points_2d, K, R, t)  # (B, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (B, V, 3)
        centers_expanded = centers[:, :, None, :].expand(B, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (B, V, J, 6)

        obs_emb = self.obs_embed(x)  # (B, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (B, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (B, V, J, d)

        # Normalized camera embedding: per-view, broadcast to all joints.
        camera_feat = _normalized_camera_embedding(K, R, t)  # (B, V, 13)
        camera_emb = self.camera_embed_mlp(camera_feat)  # (B, V, d)
        camera_emb = camera_emb[:, :, None, :].expand(B, V, J, self.d)
        feat = feat + camera_emb

        # View-level attention.
        feat_v = feat.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
        attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
        feat_v = self.view_norm1(feat_v + attn_out)
        feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))
        feat_v = feat_v.view(B, J, V, self.d)

        # Joint-level attention.
        feat_j = feat_v.permute(0, 2, 1, 3).reshape(B * V, J, self.d)
        for layer in self.joint_attn:
            feat_j = layer(feat_j)
        feat_j = feat_j.view(B, V, J, self.d).permute(0, 2, 1, 3)  # (B, J, V, d)

        feat_jv = feat_j.reshape(B, J, V * self.d)
        feat_fused = self.fusion_mlp(feat_jv)

        w_logits = self.weight_head(feat_j).squeeze(-1)  # (B, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B, V, J)
        weights = weights * confidences

        Rt = torch.cat([R, t[:, :, :, None]], dim=-1)
        P = K @ Rt
        pred_3d = self._triangulate_weighted_dlt(points_2d, weights, P)
        return pred_3d, weights

    def _triangulate_weighted_dlt(self, points_2d, weights, proj_matrices):
        from .ray_attention_model import _triangulate_weighted_dlt
        return _triangulate_weighted_dlt(points_2d, weights, proj_matrices)
