"""Cross-view spatio-temporal ray-aware fusion with residual refinement and camera PE (v2).

Integrates ``CameraPositionalEncoding`` into the best-performing cross-view
residual model.  The geometry-based camera embedding replaces the raw K/R/t MLP,
making the model invariant to absolute scene scale/focal length and easier to
transfer across camera rigs.  The rest of the architecture (joint/cross-view
attention, spatio-temporal transformer, weighted DLT, residual head) is left
unchanged.
"""

import torch

from .ray_attention_temporal_crossview_residual_model import RayAttentionFusionModelTemporalCrossviewResidual
from .camera_positional_encoding import CameraPositionalEncoding


class RayAttentionFusionModelTemporalCrossviewResidualCamPEV2(
    RayAttentionFusionModelTemporalCrossviewResidual
):
    """Cross-view residual model with geometry-based camera positional encoding.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_st_layers, max_temporal_len,
    residual_hidden:
        See ``RayAttentionFusionModelTemporalCrossviewResidual``.
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
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        n_bands: int = 4,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
        )
        # Replace the raw K/R/t MLP camera embed with geometry-based camera PE.
        self.camera_embed_mlp = CameraPositionalEncoding(d=d, n_bands=n_bands)
        self.n_bands = n_bands

    def _extract_frame_features(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run the per-frame encoder with camera PE.  Input (N, V, J, 3); output (N, V, J, d)."""
        N, V, J, _ = x.shape
        from .ray_attention_model import _compute_rays

        points_2d = x[..., :2]
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


if __name__ == "__main__":
    import torch
    from .ray_attention_temporal_crossview_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalCrossviewResidualCamPEV2(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Cross-view CamPE v2 sanity check passed (4 views)")
