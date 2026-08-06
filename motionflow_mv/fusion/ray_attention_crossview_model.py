"""Cross-view transformer ray-aware attention fusion.

Extends `RayAttentionFusionModelTemporal` by adding an explicit pairwise
cross-view attention block inside the per-frame encoder.  For every joint, the
block builds a V x V grid of pairwise view tokens, runs self-attention over all
V^2 tokens, and pools back to per-view features.  The rest of the architecture
(temporal transformer + per-frame DLT triangulation) is unchanged.

Architecture:
1. Per-frame v3 encoder: observation + ray embeddings, camera-conditioned
   embeddings, view-level self-attention.
2. Pairwise cross-view attention: for each (source, target) view pair a token is
   formed, a learned pair positional embedding is added, and transformer layers
   attend over V^2 tokens.  Tokens are then max-pooled across the source-view
   dimension to yield refined per-view features.
3. Joint-level self-attention (same as v3/temporal).
4. Temporal transformer across frames (same as temporal baseline).
5. Per-frame weighted DLT triangulation.

The pairwise block is intentionally small (one cross-view encoder layer and a
single pooling step) so the model remains lightweight while explicitly modeling
inter-view relationships.
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


class CrossViewPairAttention(nn.Module):
    """Explicit pairwise cross-view attention over V x V view pairs.

    The cross-view block uses a smaller inner dimension to keep memory modest,
    then projects back to the main feature dimension so the rest of the network
    is unchanged.

    Input:
        feat: (N, V, J, d) per-view per-joint features.
    Output:
        feat: (N, V, J, d) refined per-view features with cross-view context.
    """

    def __init__(self, n_views: int, d: int, n_heads: int = 4, n_layers: int = 1, d_cross: int = 32):
        super().__init__()
        self.n_views = n_views
        self.d = d
        self.d_cross = d_cross
        # Down-project before building pairs.
        self.down = nn.Linear(d, d_cross)
        # Pairwise combination of two view features.
        self.pair_proj = nn.Linear(d_cross * 2, d_cross)
        # Learnable positional embedding for each (source, target) pair.
        self.pair_pos_embed = nn.Parameter(torch.randn(n_views, n_views, d_cross) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_cross,
                    nhead=n_heads,
                    dim_feedforward=d_cross * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        # Up-project back to the main dimension for the residual.
        self.up = nn.Linear(d_cross, d)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        N, V, J, d = feat.shape
        # Down-project.
        feat_c = self.down(feat)  # (N, V, J, d_cross)
        # Build pairwise features: for each (source, target) view pair per joint.
        feat_src = feat_c[:, :, None, :, :].expand(N, V, V, J, self.d_cross)
        feat_tgt = feat_c[:, None, :, :, :].expand(N, V, V, J, self.d_cross)
        pair = torch.cat([feat_src, feat_tgt], dim=-1)  # (N, V, V, J, 2*d_cross)
        pair = self.pair_proj(pair)  # (N, V, V, J, d_cross)
        pair = pair + self.pair_pos_embed[None, :, :, None, :]
        # Flatten V*V and J into the batch dimension; sequence = V*V per joint.
        pair = pair.permute(0, 3, 1, 2, 4).reshape(N * J, V * V, self.d_cross)
        for layer in self.layers:
            pair = layer(pair)
        pair = pair.view(N, J, V, V, self.d_cross).permute(0, 2, 3, 1, 4)  # (N, V, V, J, d_cross)
        # Max-pool over the source view dimension to get per-target-view features.
        pooled, _ = pair.max(dim=1)  # (N, V, J, d_cross)
        pooled = self.up(pooled)
        # Residual connection with the original features.
        return feat + pooled


class RayAttentionFusionModelCrossView(nn.Module):
    """Ray-aware attention fusion with explicit pairwise cross-view attention.

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
        n_crossview_layers: int = 1,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        d_cross: int = 32,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        # Per-frame embeddings (same as v3 / temporal).
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

        # Explicit pairwise cross-view attention.
        self.crossview_attn = CrossViewPairAttention(
            n_views=n_views, d=d, n_heads=n_heads, n_layers=n_crossview_layers, d_cross=d_cross
        )

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
        """Run the per-frame encoder.  Input shape (N, V, J, 3); output (N, V, J, d)."""
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
        feat_v = feat_v.view(N, J, V, self.d).permute(0, 2, 1, 3)  # (N, V, J, d)

        # Pairwise cross-view attention.
        feat_cv = self.crossview_attn(feat_v)

        # Joint-level attention.
        feat_j = feat_cv.permute(0, 2, 1, 3).reshape(N * V, J, self.d)
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

        # Per-frame features with cross-view attention.
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Temporal transformer over frames.
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
