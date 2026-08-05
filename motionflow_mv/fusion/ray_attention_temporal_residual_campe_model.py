"""Temporal ray-aware attention fusion with residual refinement and camera PE.

Replaces the learned camera-embedding MLP in
``RayAttentionFusionModelTemporalResidual`` with a geometry-based camera
positional encoding.  This makes the model independent of the number of views
and improves cross-dataset transfer.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .camera_positional_encoding import CameraPositionalEncoding
from ..calibration.camera import Camera


class RayAttentionFusionModelTemporalResidualCamPE(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-aware fusion with residual refinement and camera PE.

    This subclass overrides the per-frame feature extractor to inject a
    geometry-based camera positional encoding instead of the original
    camera-embedding MLP.  All other components (ray embedding, view/joint
    attention, temporal attention, weight head, and residual head) are reused
    unchanged, so checkpoints of the original model remain structurally
    compatible except for the camera-embedding parameters.

    Parameters
    ----------
    See ``RayAttentionFusionModelTemporalResidual``.  Additional parameters:

    n_bands:
        Number of Fourier bands for camera positional encoding.
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
        residual_hidden: int = 128,
        n_bands: int = 4,
    ):
        # Initialize the parent with the same hyper-parameters.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
        )
        # Replace the original camera-embedding MLP with camera PE.
        self.camera_embed_mlp = CameraPositionalEncoding(d=d, n_bands=n_bands)
        # Remove the unused view-dependent fusion_mlp so the model can accept
        # different numbers of views at inference time.
        del self.fusion_mlp

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the v3 per-frame encoder with camera PE.

        Input shape (N, V, J, 3); output (N, V, J, d).
        """
        from .ray_attention_model import _compute_rays

        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        # Ray embedding.
        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)
        obs_emb = self.obs_embed(x)  # (N, V, J, d/2)
        ray_emb = self.ray_embed(torch.cat([
            torch.zeros_like(rays),  # center placeholder; kept for compatibility
            rays,
        ], dim=-1))  # (N, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (N, V, J, d)

        # Camera positional encoding.
        camera_emb = self.camera_embed_mlp(K, R, t)  # (N, V, d)
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

        return feat_j


if __name__ == "__main__":
    from .ray_attention_temporal_residual_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPE(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Camera PE temporal residual model sanity check passed")
