"""Single-frame V4 ray-aware attention fusion with residual refinement.

Extends ``RayAttentionFusionModelV4`` by adding a lightweight residual
refinement head on top of the raw DLT triangulated 3D pose.  The head
consumes the fused per-joint feature produced by the v4 encoder and
predicts a per-joint residual correction :math:`\\Delta X` that is added
back to the raw estimate.

Input:
    x: (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    X: (B, J, 3) refined world-coordinate 3D joints
    weights: (B, V, J) predicted per-view per-joint weights
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_v4_model import RayAttentionFusionModelV4, _cameras_to_tensors
from .ray_attention_model import _compute_rays
from ..calibration.camera import Camera


class RayAttentionFusionModelV4Residual(RayAttentionFusionModelV4):
    """V4 ray-aware attention fusion with a residual refinement head.

    The model predicts per-view weights, triangulates a raw 3D pose, and then
    refines it with a small MLP head that predicts per-joint residuals from the
    fused per-joint features and the raw triangulated estimate.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers:
        See ``RayAttentionFusionModelV4``.
    residual_hidden:
        Hidden dimension of the residual MLP (default 128).
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        residual_hidden: int = 128,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
        )
        self.residual_hidden = residual_hidden

        # Residual refinement head: fused per-joint feature + raw 3D joint.
        self.residual_mlp = nn.Sequential(
            nn.Linear(d + 3, residual_hidden),
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
        from .ray_attention_v4_model import _normalized_camera_embedding

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
        feat_fused = self.fusion_mlp(feat_jv)  # (B, J, d)

        w_logits = self.weight_head(feat_j).squeeze(-1)  # (B, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B, V, J)
        weights = weights * confidences

        Rt = torch.cat([R, t[:, :, :, None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = self._triangulate_weighted_dlt(points_2d, weights, P)  # (B, J, 3)

        # Residual refinement head.
        residual_input = torch.cat([feat_fused, pred_3d_raw], dim=-1)  # (B, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B, J, 3)
        pred_3d = pred_3d_raw + delta

        return pred_3d, weights
