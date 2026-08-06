"""Temporal ray-aware fusion with residual refinement and camera PE (v2).

This is a refined integration of ``CameraPositionalEncoding`` into the main
residual model (``RayAttentionFusionModelTemporalResidual``).  Compared with the
first CamPE variant, v2:

* keeps the full ray embedding (camera center + ray direction) instead of a
  zero centre placeholder;
* removes the unused ``fusion_mlp`` so the model is truly variable-view;
* exposes the number of Fourier bands as a constructor argument.

Otherwise the architecture is identical to the main residual model: view-level
and joint-level self-attention, temporal transformer, weighted DLT, and a
residual refinement head.
"""

from typing import List

import torch

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .camera_positional_encoding import CameraPositionalEncoding
from .ray_attention_model import _compute_rays
from ..calibration.camera import Camera


class RayAttentionFusionModelTemporalResidualCamPEV2(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-aware fusion with residual refinement and geometry-based camera PE.

    The camera-conditioned embedding is replaced by ``CameraPositionalEncoding``,
    which derives per-view tokens from calibrated intrinsics/extrinsics.  This
    removes the dependency on a fixed ``n_views`` ordering and makes the model
    invariant to absolute scene scale and focal length.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_temporal_layers, max_temporal_len,
    residual_hidden, use_reproj_gate:
        See ``RayAttentionFusionModelTemporalResidual``.
    n_bands:
        Number of Fourier bands used by the camera positional encoding.
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
        use_reproj_gate: bool = False,
        n_bands: int = 4,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            use_reproj_gate=use_reproj_gate,
        )
        # Replace the raw K/R/t MLP camera embed with geometry-based camera PE.
        self.camera_embed_mlp = CameraPositionalEncoding(d=d, n_bands=n_bands)
        self.n_bands = n_bands

        # The fusion_mlp is tied to a fixed number of views and is not used by
        # the residual model.  Remove it so the model can accept different V.
        if hasattr(self, "fusion_mlp"):
            del self.fusion_mlp

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the per-frame encoder with camera PE.

        Input shape (N, V, J, 3); output (N, V, J, d).
        """
        N, V, J, _ = x.shape
        points_2d = x[..., :2]

        # Ray embedding with camera centres (same as the base residual model).
        rays = _compute_rays(points_2d, K, R, t)  # (N, V, J, 3)
        centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (N, V, 3)
        centers_expanded = centers[:, :, None, :].expand(N, V, J, 3)
        ray_input = torch.cat([centers_expanded, rays], dim=-1)  # (N, V, J, 6)

        obs_emb = self.obs_embed(x)  # (N, V, J, d/2)
        ray_emb = self.ray_embed(ray_input)  # (N, V, J, d/2)
        feat = torch.cat([obs_emb, ray_emb], dim=-1)  # (N, V, J, d)

        # Geometry-based camera positional encoding.
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

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
    ):
        # Re-use the parent's full temporal+residual forward.  This method is
        # only here to keep the signature discoverable and to document that the
        # model is fully compatible with the base residual forward.
        return super().forward(
            x=x,
            cameras=cameras,
            K=K,
            R=R,
            t=t,
            n_iter=n_iter,
        )


if __name__ == "__main__":
    from .ray_attention_temporal_residual_model import _make_cameras

    # Shape/gradient sanity check with a 4-view rig.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualCamPEV2(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("CamPE v2 residual model sanity check passed (4 views)")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, w4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert w4.shape == (B, V, J)
    print("CamPE v2 single-frame compatibility passed")

    # Variable-view sanity check: same model instance run with 14 views.
    V14 = 14
    cameras_14 = _make_cameras(V14)
    x14 = torch.rand(B, T, V14, J, 3)
    pred14, w14 = model(x14, cameras=cameras_14)
    assert pred14.shape == (B, T, J, 3)
    assert w14.shape == (B, T, V14, J)
    print("CamPE v2 variable-view sanity check passed (14 views)")
