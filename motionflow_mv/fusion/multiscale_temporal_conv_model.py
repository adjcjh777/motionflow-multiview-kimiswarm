"""Ray-aware attention fusion with a multi-scale temporal convolution head.

This module mirrors `RayAttentionFusionModelTemporal` but replaces the
transformer encoder over time with a stack of multi-scale 1-D convolutional
temporal blocks.  Each block mixes information across time using several
parallel convolutions with different kernel sizes / dilations; the outputs are
concatenated and projected back to the model dimension, with a residual
connection and layer normalization.

The per-frame view/joint attention encoder is kept identical to the baseline
(v3 encoder) so that any difference in performance can be attributed to the
temporal modelling choice.
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


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale 1-D temporal convolution block.

    Input / output: ``(N, T, d)``.
    For each channel (one view-joint pair), a set of parallel Conv1d layers
    with different kernel sizes/dilations captures multi-scale temporal context.
    Their features are concatenated and linearly projected back to ``d``.
    Padding is chosen so that the temporal length is preserved.
    """

    def __init__(
        self,
        d: int,
        kernel_sizes: List[int] = None,
        dilations: List[int] = None,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        if (kernel_sizes is None) != (dilations is None):
            raise ValueError("kernel_sizes and dilations must be provided together")
        if kernel_sizes is None:
            # A small default multi-scale receptive field set.
            kernel_sizes = [3, 3, 3]
            dilations = [1, 2, 4]
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")

        self.d = d
        self.branches = nn.ModuleList()
        for k, dil in zip(kernel_sizes, dilations):
            # Symmetric "same" padding for odd kernels.
            pad = (k - 1) * dil // 2
            self.branches.append(
                nn.Conv1d(d, d, kernel_size=k, dilation=dil, padding=pad, bias=False)
            )
        self.proj = nn.Linear(d * len(self.branches), d)
        self.norm = nn.LayerNorm(d)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d)
        x_in = x
        x_c = x.transpose(1, 2)  # (N, d, T)
        outs = [branch(x_c) for branch in self.branches]  # each (N, d, T)
        x_cat = torch.cat(outs, dim=1)  # (N, branches*d, T)
        x_out = self.proj(x_cat.transpose(1, 2))  # (N, T, d)
        x_out = self.activation(self.norm(x_out + x_in))
        return x_out


class MultiScaleTemporalConvModel(nn.Module):
    """Ray-aware multi-view fusion with multi-scale temporal convolutions.

    The model consumes a clip of multi-view 2D keypoints
    ``(B, T, V, J, 3)`` and predicts per-frame 3D poses ``(B, T, J, 3)``
    plus per-view weights ``(B, T, V, J)``.  A 4D input ``(B, V, J, 3)``
    is treated as a single-frame clip for convenience.

    Parameters
    ----------
    j: number of joints (default 17).
    d: feature dimension (default 64).
    n_views: number of calibrated views.
    n_heads: number of attention heads for the per-frame encoder.
    n_joint_layers: number of joint-level transformer layers.
    n_temporal_layers: number of multi-scale temporal convolution blocks.
    max_temporal_len: maximum temporal length for position embeddings.
    temporal_kernel_sizes, temporal_dilations: lists that configure the
        parallel convolutions in every temporal block.  If ``None``, a default
        multi-scale configuration (kernel 3 with dilations [1, 2, 4]) is used.
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
        temporal_kernel_sizes: List[int] = None,
        temporal_dilations: List[int] = None,
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

        # Temporal position embedding and multi-scale temporal conv blocks.
        self.temporal_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.temporal_blocks = nn.ModuleList(
            [
                MultiScaleTemporalBlock(
                    d,
                    kernel_sizes=temporal_kernel_sizes,
                    dilations=temporal_dilations,
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
        for block in self.temporal_blocks:
            feat = block(feat)
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
