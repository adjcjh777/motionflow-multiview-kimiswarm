"""Temporal ray-aware attention fusion with v4 normalized camera embeddings.

Extends ``RayAttentionFusionModelV4`` by stacking a temporal transformer on top
of its per-frame view/joint attention encoder.  Module names for the per-frame
encoder are kept identical to the single-frame V4 model so a pretrained V4
checkpoint can be loaded directly into the temporal variant (the temporal layers
are initialized from scratch).

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    X: (B, T, J, 3) world-coordinate 3D joints, or (B, J, 3) for 4D input
    weights: (B, T, V, J) predicted per-view per-joint weights, or (B, V, J)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_model import _compute_rays, _triangulate_weighted_dlt
from .ray_attention_v4_model import _normalized_camera_embedding
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelTemporalV4(nn.Module):
    """Ray-aware temporal fusion model using the V4 per-frame encoder."""

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        # Per-frame embeddings (identical names to RayAttentionFusionModelV4).
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

        # Fusion MLP and weight head.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.weight_head = nn.Linear(d, 1)

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the V4 per-frame encoder.  Input shape (N, V, J, 3); output (N, V, J, d)."""
        N, V, J, _ = x.shape

        points_2d = x[..., :2]

        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (N, V, 3)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (N, V, J, 6)

        obs_emb = self.obs_embed(x)  # (N, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (N, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (N, V, J, d)

        # Normalized camera embedding: per-view, broadcast to all joints.
        camera_feat = _normalized_camera_embedding(K, R, t)  # (N, V, 13)
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

    def load_v4_state_dict(self, state_dict, strict: bool = False):
        """Load weights from a RayAttentionFusionModelV4 checkpoint.

        Only keys that exist in both state_dicts are copied; temporal layers are
        left at their random initialization.  Set ``strict=True`` to require an
        exact match for the loaded subset.
        """
        own_state = self.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in own_state}
        if strict and len(filtered) != len([k for k in own_state if not k.startswith("temporal")]):
            missing = [k for k in own_state if k not in state_dict and not k.startswith("temporal")]
            raise RuntimeError(f"Missing expected V4 keys when loading: {missing}")
        self.load_state_dict(filtered, strict=False)

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
            # Single rig: (V, ...) -> (B*T, V, ...)
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            # Per-batch rig: (B, V, ...) -> (B*T, V, ...)
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Per-frame V4 features. Clone camera tensors to avoid view/stride
        # conflicts between the per-frame encoder and the later DLT stage.
        feat = self._extract_frame_features(x_flat, K.clone(), R.clone(), t.clone())  # (B*T, V, J, d)

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        feat = feat.clone()
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights
