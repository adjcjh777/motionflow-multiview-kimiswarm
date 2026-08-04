"""Temporal ray-attention residual model with a V4-normalized per-frame encoder.

This extends ``RayAttentionFusionModelTemporalResidual`` by replacing the raw
21-D camera embedding with the normalized -D camera embedding from
``RayAttentionFusionModelV4``.  The per-frame encoder therefore becomes
bit-for-bit compatible with a V4 checkpoint, enabling the two-stage training
curriculum:

    1. Pre-train ``RayAttentionFusionModelV4`` on single-frame MPI-INF-3DHP.
    2. Load the V4 checkpoint into this model's per-frame encoder.
    3. Fine-tune the full temporal+residual model on video clips.

All temporal and residual parameters remain randomly initialised and are
learnt during fine-tuning.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .ray_attention_v4_model import _normalized_camera_embedding
from .ray_attention_model import _compute_rays


class RayAttentionFusionModelTemporalResidualV2(RayAttentionFusionModelTemporalResidual):
    """Temporal residual model whose per-frame encoder matches V4.

    See ``RayAttentionFusionModelTemporalResidual`` for the full architecture.
    The only difference is the camera-conditioned embedding, which uses the
    scale/normalisation invariant 13-D camera feature from V4.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the raw K/R/t camera embed with the V4 normalised one.
        self.camera_embed_mlp = nn.Sequential(
            nn.Linear(13, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )

    def _extract_frame_features(self, x, K, R, t):
        """Run the V4 per-frame encoder.  Input (N, V, J, 3); output (N, V, J, d)."""
        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (N, V, 3)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (N, V, J, 6)

        obs_emb = self.obs_embed(x)  # (N, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (N, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (N, V, J, d)

        # V4 normalised camera embedding: per-view, broadcast to all joints.
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
