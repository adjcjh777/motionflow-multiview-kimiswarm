"""Factorised (T x V x J) spatio-temporal transformer with principal-point input.

The model consumes per-view 2D keypoints ``x: (B, T, V, J, 3)`` together with
calibrated cameras ``(K, R, t)``, corrects the principal point of each view,
encodes every frame into per-token features, then refines the tokens with three
factorised attention axes (temporal, view, joint), and finally triangulates 3D
joints with learned per-view weights and a residual refinement head.

Input / output semantics are compatible with the existing
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` family: a
4-D input ``(B, V, J, 3)`` is treated as a single-frame clip.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from motionflow_mv.calibration.camera import Camera
from motionflow_mv.fusion.principal_point_correction import PrincipalPointCorrection
from motionflow_mv.fusion.ray_attention_model import (
    _compute_rays,
    _triangulate_weighted_dlt,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_model import (
    _cameras_to_tensors,
)


class SpatiotemporalPrincipalPointModel(nn.Module):
    """Factorised T-V-J transformer with learned principal-point correction.

    Parameters
    ----------
    j:
        Number of joints.
    d:
        Token dimension.
    n_views:
        Number of views (fixed-slot; all views are processed).
    n_heads:
        Number of attention heads.
    n_temporal_layers, n_view_layers, n_joint_layers:
        Number of TransformerEncoderLayers along each axis.  The default of one
        layer per axis is sufficient for a smoke-testable skeleton.
    max_temporal_len:
        Maximum sequence length used to size the temporal positional embedding.
    residual_hidden:
        Hidden dimension of the residual refinement MLP.  If ``None``, the
        residual head is omitted.
    principal_point_hidden:
        Hidden dimension of the principal-point offset MLP.
    principal_point_max_offset:
        Maximum absolute principal-point correction in pixels.
    focal_max_scale:
        If ``> 0``, also learn a per-view focal-length scale.
    return_pp_delta:
        If ``True``, also return the predicted principal-point offset.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_temporal_layers: int = 1,
        n_view_layers: int = 1,
        n_joint_layers: int = 1,
        max_temporal_len: int = 256,
        residual_hidden: Optional[int] = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.n_heads = n_heads
        self.return_pp_delta = return_pp_delta
        self.correct_focal = focal_max_scale > 0.0

        # Per-frame embeddings.
        self.obs_embed = nn.Linear(3, d // 2)
        self.ray_embed = nn.Linear(6, d // 2)

        # Camera-conditioned embedding.
        self.camera_embed_mlp = nn.Sequential(
            nn.Linear(9 + 9 + 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Principal-point / intrinsic correction layer.
        self.principal_point_correction = PrincipalPointCorrection(
            d=d,
            hidden=principal_point_hidden,
            max_offset=principal_point_max_offset,
            max_focal_scale=focal_max_scale,
        )

        # Lightweight per-frame view + joint attention (borrowed from the
        # temporal/cross-view models) before the factorised spatio-temporal
        # block.  This gives the model a strong local per-frame representation.
        self.frame_view_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, batch_first=True
        )
        self.frame_view_norm1 = nn.LayerNorm(d)
        self.frame_view_ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
        )
        self.frame_view_norm2 = nn.LayerNorm(d)

        self.frame_joint_attn = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_joint_layers)
            ]
        )

        # Positional embeddings for the three factorised axes.
        self.time_pos_embed = nn.Parameter(torch.randn(max_temporal_len, d) * 0.02)
        self.view_pos_embed = nn.Parameter(torch.randn(n_views, d) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(j, d) * 0.02)

        # Factorised attention along T, V, J.
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_temporal_layers)
            ]
        )
        self.view_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_view_layers)
            ]
        )
        self.joint_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_joint_layers)
            ]
        )

        # Weight head and optional residual refinement head.
        self.weight_head = nn.Linear(d, 1)

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
        """Run the per-frame encoder.  Input (N, V, J, 3); output (N, V, J, d)."""
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
        camera_feat = torch.cat(
            [K.view(N, V, -1), R.view(N, V, -1), t.view(N, V, -1)],
            dim=-1,
        )  # (N, V, 21)
        camera_emb = self.camera_embed_mlp(camera_feat)  # (N, V, d)
        camera_emb = camera_emb[:, :, None, :].expand(N, V, J, self.d)
        feat = feat + camera_emb

        # View-level attention.
        feat_v = feat.permute(0, 2, 1, 3).reshape(N * J, V, self.d)
        attn_out, _ = self.frame_view_attn(feat_v, feat_v, feat_v)
        feat_v = self.frame_view_norm1(feat_v + attn_out)
        feat_v = self.frame_view_norm2(feat_v + self.frame_view_ffn(feat_v))
        feat_v = feat_v.view(N, J, V, self.d)

        # Joint-level attention.
        feat_j = feat_v.permute(0, 2, 1, 3).reshape(N * V, J, self.d)
        for layer in self.frame_joint_attn:
            feat_j = layer(feat_j)
        feat_j = feat_j.view(N, V, J, self.d)

        return feat_j

    def _factorised_attention(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply factorised attention along T, then V, then J.

        Input:  (B, T, V, J, d)
        Output: (B, T, V, J, d)
        """
        B, T, V, J, _ = feat.shape

        # Temporal axis: attend over T for each (view, joint).
        if self.temporal_layers:
            feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
            for layer in self.temporal_layers:
                feat = layer(feat)
            feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4)


        # View axis: attend over V for each (time, joint).
        if self.view_layers:
            feat = feat.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, self.d)
            for layer in self.view_layers:
                feat = layer(feat)
            feat = feat.view(B, T, J, V, self.d).permute(0, 1, 3, 2, 4)

        # Joint axis: attend over J for each (time, view).
        if self.joint_layers:
            feat = feat.reshape(B * T * V, J, self.d)
            for layer in self.joint_layers:
                feat = layer(feat)
            feat = feat.view(B, T, V, J, self.d)

        return feat

    def forward(
        self,
        x: torch.Tensor,
        cameras: Optional[List[Camera]] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Broadcast camera rig to (B*T, V, ...).
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]  # (B*T, V, J)

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame features.
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Reshape to spatio-temporal grid and add 3-D positional embeddings.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        joint_emb = self.joint_pos_embed[:J].view(1, 1, 1, J, self.d)
        feat = feat + time_emb + view_emb + joint_emb

        # Factorised (T x V x J) attention.
        feat = self._factorised_attention(feat)

        # Per-view weight prediction.
        feat_for_weight = feat.permute(0, 2, 1, 3, 4).reshape(B * T, V, J, self.d)
        feat_for_weight = feat_for_weight.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        # Triangulate with corrected intrinsics.
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Optional residual refinement head.
        if self.residual_mlp is not None:
            feat_pooled = feat.mean(dim=2)  # average over views -> (B, T, J, d)
            feat_pooled = feat_pooled.reshape(B * T, J, self.d)
            residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
            delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
            pred_3d = pred_3d_raw + delta
        else:
            pred_3d = pred_3d_raw

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.return_pp_delta:
            pp_delta = pp_delta.view(B, T, V, 2)
            if self.correct_focal:
                focal_scale = focal_scale.view(B, T, V)
                if squeeze_output:
                    pp_delta = pp_delta.squeeze(1)
                    focal_scale = focal_scale.squeeze(1)
                return pred_3d, weights, pp_delta, focal_scale
            if squeeze_output:
                pp_delta = pp_delta.squeeze(1)
            return pred_3d, weights, pp_delta

        return pred_3d, weights


def _make_cameras(n_views: int = 4) -> List[Camera]:
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
    model = SpatiotemporalPrincipalPointModel(
        j=J,
        d=64,
        n_views=V,
        return_pp_delta=True,
    )
    pred, w, pp_delta = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    assert pp_delta.shape == (B, T, V, 2)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    total = sum(p.numel() for p in model.parameters())
    print(f"SpatiotemporalPrincipalPointModel parameters: {total:,}")
    print("spatio-temporal principal-point model sanity check passed")
