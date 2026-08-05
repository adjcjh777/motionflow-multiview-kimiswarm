"""Ray-aware attention fusion v3: camera-conditioned embeddings + domain-agnostic camera features.

Adds a learned camera embedding derived from intrinsics and extrinsics to the
per-view tokens before attention. The camera feature vector is now normalized
by focal length and rig diameter so the model generalizes across camera rigs
with different absolute scales.

Optionally supports a gradient-reversal domain classifier on the pooled camera
embedding. When ``use_domain_classifier=True`` and ``domain_labels`` are passed,
the forward pass returns ``(pred_3d, weights, domain_logits)`` and the GRL
can be trained adversarially to make camera embeddings domain-invariant.

Summary of changes (swarm iter5):
- ``_domain_agnostic_camera_features`` normalizes K by focal length and camera
centers by rig diameter.
- ``GradientReversalLayer`` + optional ``domain_classifier`` are added but
off by default, preserving two-output compatibility with existing trainers.
- Small verification forward runs on H36M s_01 (source) and s_09 (target)
confirm the model runs end-to-end; cross-dataset MPJPE numbers require a
finetuned checkpoint (not run here per no-long-training instruction).
"""

from typing import List, Tuple, Optional

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


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer wrapper with scalar lambda."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


def _domain_agnostic_camera_features(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Build a domain-agnostic camera descriptor of the same 21-D size as raw K,R,t.

    Normalization:
        - K is divided by the geometric mean focal length (fx*fy)^0.5.
        - Camera centers c = -R^T t are normalized by the rig diameter so
          different camera baselines/scales are comparable across datasets.

    Args:
        K: (B, V, 3, 3) intrinsics.
        R: (B, V, 3, 3) rotations.
        t: (B, V, 3) translation.

    Returns:
        (B, V, 21) tensor [flattened K_norm, flattened R, normalized c].
    """
    B, V, _, _ = K.shape

    # Normalize intrinsics by focal length.
    fx = K[:, :, 0, 0]
    fy = K[:, :, 1, 1]
    f = torch.sqrt(fx * fy + 1e-8)
    K_norm = K / f[:, :, None, None]

    # Camera centers in world coordinates.
    centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (B, V, 3)
    centered = centers - centers.mean(dim=1, keepdim=True)
    scale = centered.norm(dim=-1).max(dim=1, keepdim=True)[0] + 1e-6
    centers_norm = centers / scale.unsqueeze(-1)

    camera_feat = torch.cat(
        [K_norm.view(B, V, -1), R.view(B, V, -1), centers_norm],
        dim=-1,
    )
    return camera_feat


class RayAttentionFusionModelV3(nn.Module):
    """Ray-aware attention fusion with camera embeddings, view attention, and cross-joint attention.

    Args:
        j: number of joints.
        d: hidden dimension.
        n_views: number of views.
        n_heads: attention heads.
        n_joint_layers: number of joint-level transformer layers.
        use_domain_classifier: if True, add a gradient-reversal domain classifier.
        n_domains: number of source/target domains for the classifier.
        grl_lambda: gradient-reversal scaling factor.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        use_domain_classifier: bool = False,
        n_domains: int = 2,
        grl_lambda: float = 1.0,
        use_camera_emb: bool = True,
        use_view_attn: bool = True,
        use_joint_attn: bool = True,
        direct_regression: bool = False,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads
        self.use_camera_emb = use_camera_emb
        self.use_view_attn = use_view_attn
        self.use_joint_attn = use_joint_attn
        self.direct_regression = direct_regression
        self.use_domain_classifier = use_domain_classifier
        self.n_domains = n_domains

        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        # Camera embedding: flatten K, R, t -> d (domain-agnostic features are fed here).
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
            [nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True, norm_first=True)
             for _ in range(n_joint_layers)]
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d * n_views, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        self.weight_head = nn.Linear(d, 1)

        if self.direct_regression:
            self.direct_head = nn.Linear(d, 3)

        # Optional domain-adversarial head on pooled camera embeddings.
        if self.use_domain_classifier:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, n_domains),
            )
        else:
            self.grl = None
            self.domain_classifier = None

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
        return_domain_logits: bool = False,
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

        # Domain-agnostic camera embedding: per-view, broadcast to all joints.
        if self.use_camera_emb:
            camera_feat = _domain_agnostic_camera_features(K, R, t)  # (B, V, 21)
            camera_emb = self.camera_embed_mlp(camera_feat)  # (B, V, d)
            camera_emb_expanded = camera_emb[:, :, None, :].expand(B, V, J, self.d)
            feat = feat + camera_emb_expanded

        # View-level attention.
        if self.use_view_attn:
            feat_v = feat.permute(0, 2, 1, 3).reshape(B * J, V, self.d)
            attn_out, _ = self.view_attn(feat_v, feat_v, feat_v)
            feat_v = self.view_norm1(feat_v + attn_out)
            feat_v = self.view_norm2(feat_v + self.view_ffn(feat_v))
            feat_v = feat_v.view(B, J, V, self.d)
        else:
            feat_v = feat.permute(0, 2, 1, 3)  # (B, J, V, d)

        # Joint-level attention.
        if self.use_joint_attn:
            feat_j = feat_v.permute(0, 2, 1, 3).reshape(B * V, J, self.d)
            for layer in self.joint_attn:
                feat_j = layer(feat_j)
            feat_j = feat_j.view(B, V, J, self.d).permute(0, 2, 1, 3)  # (B, J, V, d)
        else:
            feat_j = feat_v

        feat_jv = feat_j.reshape(B, J, V * self.d)
        feat_fused = self.fusion_mlp(feat_jv)

        w_logits = self.weight_head(feat_j).squeeze(-1)  # (B, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B, V, J)
        weights = weights * confidences

        if self.direct_regression:
            pred_3d = self.direct_head(feat_fused)
        else:
            Rt = torch.cat([R, t[:, :, :, None]], dim=-1)
            P = K @ Rt
            pred_3d = self._triangulate_weighted_dlt(points_2d, weights, P)

        # Optional domain-adversarial output.
        if self.use_domain_classifier and (domain_labels is not None or return_domain_logits):
            domain_logits = self.domain_classifier(self.grl(camera_emb.mean(dim=1)))  # (B, n_domains)
            return pred_3d, weights, domain_logits

        return pred_3d, weights

    def _triangulate_weighted_dlt(self, points_2d, weights, proj_matrices):
        from .ray_attention_model import _triangulate_weighted_dlt
        return _triangulate_weighted_dlt(points_2d, weights, proj_matrices)
