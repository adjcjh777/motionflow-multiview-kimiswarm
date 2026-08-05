"""Temporal ray-aware attention fusion.

Extends `RayAttentionFusionModelV3` by stacking a temporal transformer on top of
its per-frame view/joint attention encoder.  The model consumes a clip of
multi-view 2D keypoints (B, T, V, J, 3) and predicts per-frame 3D poses
(B, T, J, 3) plus per-view weights (B, T, V, J).  A 4D input (B, V, J, 3) is
treated as a single-frame clip for convenience.

Architecture (kept simple):
1. Per-frame v3 encoder: observation + ray embeddings, camera-conditioned
   embeddings, view-level self-attention, joint-level self-attention.
2. Temporal transformer: the per-frame (V, J) feature grid is reshaped into a
   sequence of V*J tokens per frame; self-attention is applied over the T
   frames.  A learned temporal position embedding encodes frame order.
3. Per-frame weighted DLT triangulation using the temporally-refined
   per-view weights.

Verification:
- `tests/test_ray_attention_temporal.py` runs a synthetic forward/backward
  sanity check.  It confirms that the model produces the expected output shapes,
  that gradients flow through the temporal transformer, and that a 4D input
  collapses to a 3D output as expected.

Important findings / notes:
- The temporal layer increases model capacity, so it should only be trained on
  real video clips; the synthetic single-frame pipeline used for v1/v2/v3 is
  not enough to exploit temporal context.
- The per-frame v3 encoder is duplicated here rather than refactored, because
  v3 does not expose its intermediate features.  This keeps changes local to
  this file and avoids modifying the existing v3 checkpoint/behaviour.
- A lightweight wrapper / FusionModule plugin and a small training script are
  still TODO; they are not needed to validate the model itself.
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


class RayAttentionFusionModelTemporal(nn.Module):
    """Ray-aware attention fusion with view attention, joint attention, and temporal attention.

    Input:
        x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
        cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
        OR
        K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

    Output:
        X: (B, T, J, 3) world-coordinate 3D joints, or (B, J, 3) for 4D input
        weights: (B, T, V, J) predicted per-view per-joint weights, or (B, V, J)
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
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        # Per-frame embeddings (same as v3).
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

        # Per-frame v3 features.
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
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


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras
