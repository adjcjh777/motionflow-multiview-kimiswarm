"""Cross-view spatio-temporal ray-aware attention fusion.

This model treats the multi-view pose estimation problem as a single 3-D grid of
spatio-temporal tokens.  Each token corresponds to one (time, view, joint)
tuple, and a transformer encoder attends jointly over all three axes.  This is
different from the temporal model (attends only over time for each view/joint
pair) and the cross-view temporal model (attends over time+views for each joint):
here joints, views, and time steps can all interact in the same attention field.

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
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelSpatiotemporal(nn.Module):
    """Ray-aware fusion with a unified spatio-temporal (T x V x J) transformer.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_st_layers, max_temporal_len:
        See notes below.
    residual_hidden:
        Hidden dimension of the optional residual refinement MLP.  If ``None``,
        no residual head is used.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int | None = 128,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        # Per-frame embeddings (same as v3/temporal).
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

        # 3-D positional embeddings for the unified spatio-temporal grid.
        self.time_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.view_pos_embed = nn.Parameter(torch.randn(n_views, d) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(j, d) * 0.02)

        # Unified spatio-temporal transformer over (time, view, joint) tokens.
        self.st_transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True
                )
                for _ in range(n_st_layers)
            ]
        )

        # Fusion MLP and weight head.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.weight_head = nn.Linear(d, 1)

        # Optional residual refinement head.
        self.residual_hidden = residual_hidden
        if residual_hidden is not None:
            self.residual_mlp = nn.Sequential(
                nn.Linear(d + 3, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, 3),
            )
        else:
            self.residual_mlp = None

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the per-frame encoder. Input shape (N, V, J, 3); output (N, V, J, d)."""
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

        # Prepare per-sample camera tensors and flatten time for the per-frame encoder.
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
        confidences = x_flat[..., 2]  # (B*T, V, J)

        # Per-frame features.
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Reshape to spatio-temporal grid and add 3-D positional embeddings.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        joint_emb = self.joint_pos_embed[:J].view(1, 1, 1, J, self.d)
        feat = feat + time_emb + view_emb + joint_emb

        # Flatten (T, V, J) tokens and apply unified spatio-temporal attention.
        # Each token can attend to every other token across time, views, and joints.
        feat = feat.permute(0, 4, 1, 2, 3).reshape(B, self.d, T * V * J).permute(0, 2, 1)  # (B, T*V*J, d)
        for layer in self.st_transformer:
            feat = layer(feat)

        # Reshape back to per-frame features.
        feat = feat.permute(0, 2, 1).reshape(B, self.d, T, V, J).permute(0, 2, 3, 4, 1).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Optional residual refinement head.
        if self.residual_mlp is not None:
            feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)  # (B*T, J, d+3)
            delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
            pred_3d = pred_3d + delta

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


if __name__ == "__main__":
    # Quick shape/gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelSpatiotemporal(j=J, d=64, n_views=V, n_st_layers=2)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("spatio-temporal ray-attention model sanity check passed")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, w4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert w4.shape == (B, V, J)
    print("single-frame compatibility sanity check passed")
