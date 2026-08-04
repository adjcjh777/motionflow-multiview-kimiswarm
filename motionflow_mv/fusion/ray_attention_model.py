"""Ray-aware attention fusion for calibrated multi-view pose.

The network consumes per-view 2D keypoints, confidences and calibrated cameras,
computes ray directions and camera centers, and predicts per-view weights that
are fed into a differentiable weighted DLT triangulator.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> tuple:
    """Convert list of Camera objects to (K, R, t, P) tensors on device.

    Returns:
        K: (V, 3, 3)
        R: (V, 3, 3)
        t: (V, 3)
        P: (V, 3, 4)
    """
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    P = torch.cat([R, t[:, :, None]], dim=-1)  # (V, 3, 4)
    P = K @ P
    return K, R, t, P


def _compute_rays(
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Compute camera centers and ray directions from 2D points and cameras.

    Args:
        points_2d: (B, V, J, 2)
        K: (V, 3, 3)
        R: (V, 3, 3)
        t: (V, 3)

    Returns:
        centers: (V, 3) camera centers c_v = -R_v^T t_v
        rays: (B, V, J, 3) normalized ray directions in world coordinates
    """
    # Camera centers: c = -R^T t
    centers = -torch.einsum("vij,vj->vi", R.transpose(-2, -1), t)  # (V, 3)

    # Back-project image points to camera rays and rotate to world coordinates.
    # In camera coordinates: d_cam = K^{-1} [x, y, 1]^T
    # In world coordinates:  d_world = R^T d_cam
    B, V, J, _ = points_2d.shape
    ones = torch.ones(B, V, J, 1, device=points_2d.device, dtype=points_2d.dtype)
    xy1 = torch.cat([points_2d, ones], dim=-1)  # (B, V, J, 3)

    # K^{-1} @ xy1 per view
    # K: (V, 3, 3), xy1: (B, V, J, 3)
    d_cam = torch.einsum("vik,bvjk->bvji", torch.inverse(K), xy1)  # (B, V, J, 3)
    d_world = torch.einsum("vik,bvjk->bvji", R.transpose(-2, -1), d_cam)

    # Normalize
    rays = d_world / (d_world.norm(dim=-1, keepdim=True) + 1e-8)
    return centers, rays


def _triangulate_weighted_dlt(
    points_2d: torch.Tensor,
    weights: torch.Tensor,
    proj_matrices: torch.Tensor,
) -> torch.Tensor:
    """Differentiable weighted DLT triangulation.

    Args:
        points_2d: (B, V, J, 2)
        weights: (B, V, J)
        proj_matrices: (V, 3, 4)

    Returns:
        X: (B, J, 3)
    """
    B, V, J, _ = points_2d.shape
    pred_3d = []
    for j in range(J):
        p2d = points_2d[:, :, j, :]  # (B, V, 2)
        w = weights[:, :, j]  # (B, V)
        X = _triangulate_joint(p2d, w, proj_matrices)
        pred_3d.append(X)
    return torch.stack(pred_3d, dim=1)  # (B, J, 3)


def _triangulate_joint(
    points_2d: torch.Tensor,
    weights: torch.Tensor,
    proj_matrices: torch.Tensor,
) -> torch.Tensor:
    """Triangulate a single joint across views.

    Args:
        points_2d: (B, V, 2)
        weights: (B, V)
        proj_matrices: (V, 3, 4)

    Returns:
        X: (B, 3)
    """
    B, V, _ = points_2d.shape
    A = []
    for v in range(V):
        P = proj_matrices[v]  # (3, 4)
        x = points_2d[:, v, 0]  # (B,)
        y = points_2d[:, v, 1]  # (B,)
        A.append(x[:, None] * P[2:3, :] - P[0:1, :])  # (B, 4)
        A.append(y[:, None] * P[2:3, :] - P[1:2, :])
    A = torch.stack(A, dim=1)  # (B, 2V, 4)

    A3 = A[:, :, :3]  # (B, 2V, 3)
    a4 = A[:, :, 3:]  # (B, 2V, 1)
    w = weights.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * V, 1)
    Aw = A3 * torch.sqrt(w + 1e-6)
    bw = -a4 * torch.sqrt(w + 1e-6)
    X, *_ = torch.linalg.lstsq(Aw, bw)
    return X.squeeze(-1)  # (B, 3)


class RayAttentionFusionModel(nn.Module):
    """Ray-aware attention fusion with differentiable weighted triangulation.

    Input:
        x: (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
        cameras: list of Camera objects (V,)

    Output:
        X: (B, J, 3) world-coordinate 3D joints
        weights: (B, V, J) predicted per-view per-joint weights
    """

    def __init__(self, j: int = 17, d: int = 64, n_views: int = 4, n_heads: int = 4):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads

        # Embed 2D point + confidence
        self.obs_embed = nn.Linear(3, d // 2)
        # Embed ray direction (3) and camera center (3) together
        self.ray_embed = nn.Linear(6, d // 2)

        # Self-attention over views per joint
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

        # Weight head predicts per-view weight per joint
        self.weight_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, cameras: List[Camera]) -> tuple:
        # x: (B, V, J, 3)
        B, V, J, _ = x.shape
        device = x.device
        K, R, t, P = _cameras_to_tensors(cameras, device)

        # 2D points and confidence
        points_2d = x[..., :2]  # (B, V, J, 2)
        confidences = x[..., 2]  # (B, V, J)

        # Compute ray features
        centers, rays = _compute_rays(points_2d, K, R, t)  # (V,3), (B,V,J,3)
        centers_expanded = centers[None, :, None, :].expand(B, V, J, 3)  # (B,V,J,3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (B,V,J,6)

        # Embed
        obs_emb = self.obs_embed(x)  # (B,V,J,d/2)
        ray_emb = self.ray_embed(ray_input)  # (B,V,J,d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (B,V,J,d)

        # Reshape for attention: per joint, views as sequence
        feat = feat.permute(0, 2, 1, 3).reshape(B * J, V, self.d)  # (B*J, V, d)
        attn_out, _ = self.attn(feat, feat, feat)  # (B*J, V, d)
        feat = self.norm1(feat + attn_out)
        feat = self.norm2(feat + self.ffn(feat))

        # Predict per-view weights per joint
        w_logits = self.weight_head(feat).squeeze(-1)  # (B*J, V)
        weights = torch.sigmoid(w_logits).view(B, J, V).permute(0, 2, 1)  # (B, V, J)

        # Combine with observed confidences
        weights = weights * confidences  # (B, V, J)

        # Differentiable weighted DLT
        pred_3d = _triangulate_weighted_dlt(points_2d, weights, P)
        return pred_3d, weights
