"""OmniMultiViewFusion v3 — multi-scale, geometry-aware multi-view fusion.

Extends ``OmniMultiViewFusionV2`` with two new architectural components:

1. **Hierarchical multi-scale temporal/cross-view fusion** – processes spatio-
   temporal tokens at multiple temporal and joint scales, then fuses them with
   a residual connection.  This lets the model reason about coarse motion
   patterns and coarse skeleton structure while preserving fine-grained detail.
2. **Camera-conditioned / epipolar-biased cross-view attention** – replaces the
   flat time+view transformer in v2 with an geometry-aware transformer whose
   cross-view attention scores are biased by the epipolar distance between
   views, and optionally conditions the per-view tokens on the calibrated
   camera parameters.

All v2 capabilities (visibility gating, uncertainty-weighted triangulation,
graph-joint attention, principal-point correction) are preserved.

Input
    ``x``: (B, T, V, J, 3) or (B, V, J, 3) containing ``(x_pixel, y_pixel, confidence)``
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output (tuple)
    pred_3d:   (B, T, J, 3) or (B, J, 3) world-coordinate 3D joints
    weights:   (B, T, V, J) or (B, V, J) per-view per-joint fusion weights
    visibility:(B, T, V, J) or (B, V, J) predicted visibility probabilities
    covariance:(B, T, V, J, 2, 2) or (B, V, J, 2, 2) predicted 2D covariances
    epipolar_loss: scalar auxiliary loss

Notes
    The constructor signature is a strict superset of ``OmniMultiViewFusionV2``;
    existing v2 checkpoints can be loaded with ``strict=False``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.epipolar_transformer_bias import (
    EpipolarBiasedTransformerEncoderLayer,
    build_temporal_bias_from_frames,
    compute_per_frame_epipolar_bias,
)
from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    H36M_17_SYMMETRY_PAIRS,
    MPI_INF_3DHP_28_PARENTS,
    MPI_INF_3DHP_28_SYMMETRY_PAIRS,
)
from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2


class _HierarchicalMultiscaleFusion(nn.Module):
    """Hierarchical multi-scale temporal / cross-view / joint fusion.

    For each temporal scale factor ``s`` in ``scales``:

    1. Pool time from ``T`` to ``max(1, T // s)``.
    2. Pool joints from ``J`` to ``max(1, J // s)``.
    3. Apply a lightweight cross-view transformer encoder layer at the coarser
       resolution.
    4. Upsample joints and time back to the original resolution.

    The multi-scale branches are concatenated and projected back to ``d`` with a
    residual connection.

    Parameters
    ----------
    d:
        Token dimension.
    n_views:
        Number of camera views.
    scales:
        Temporal / joint downsample factors.  ``1`` means full resolution.
    n_heads:
        Attention heads for the cross-view transformer layers.
    dropout:
        Dropout applied inside each transformer layer.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        scales: Sequence[int] = (1, 2, 4),
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.scales = tuple(scales)
        if any(s < 1 for s in self.scales):
            raise ValueError("All scale factors must be >= 1")

        self.branches = nn.ModuleList()
        for _ in self.scales:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=d * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.branches.append(layer)

        # Learnable fusion of multi-scale features.
        self.fusion = nn.Sequential(
            nn.Linear(d * len(self.scales), d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        x:
            Tensor of shape ``(B, T, V, J, d)``.

        Returns
        -------
        Tensor of shape ``(B, T, V, J, d)``.
        """
        B, T, V, J, d = x.shape
        x_in = x

        scale_features = []
        for scale, layer in zip(self.scales, self.branches):
            if scale == 1:
                x_s = x  # (B, T, V, J, d)
            else:
                # Temporal pooling: (B,T,V,J,d) -> (B, T//s, V, J, d)
                t_target = max(1, T // scale)
                x_s = x.permute(0, 2, 3, 4, 1).reshape(B * V * J, d, T)
                x_s = F.adaptive_avg_pool1d(x_s, t_target)
                x_s = x_s.view(B, V, J, d, t_target).permute(0, 4, 1, 2, 3)
                # (B, t_target, V, J, d)

            t_cur = x_s.shape[1]

            # Joint pooling: (B, t_cur, V, J, d) -> (B, t_cur, V, J//s, d)
            if scale > 1:
                j_target = max(1, J // scale)
                x_s = x_s.permute(0, 1, 2, 4, 3).reshape(B * t_cur * V, d, J)
                x_s = F.adaptive_avg_pool1d(x_s, j_target)
                x_s = x_s.view(B, t_cur, V, d, j_target).permute(0, 1, 2, 4, 3)
                # (B, t_cur, V, j_target, d)

            j_cur = x_s.shape[3]

            # Cross-view attention over views.
            # (B, t_cur, V, j_cur, d) -> (B * t_cur * j_cur, V, d)
            x_s = x_s.permute(0, 1, 3, 2, 4).reshape(B * t_cur * j_cur, V, d)
            x_s = layer(x_s)

            # Reshape back to (B, t_cur, V, j_cur, d)
            x_s = x_s.view(B, t_cur, j_cur, V, d).permute(0, 1, 3, 2, 4)

            # Upsample joints back to J.
            if scale > 1:
                x_s = x_s.permute(0, 1, 3, 4, 2).reshape(B * t_cur * V, d, j_cur)
                x_s = F.interpolate(x_s, size=J, mode="linear", align_corners=False)
                x_s = x_s.view(B, t_cur, V, d, J).permute(0, 1, 2, 4, 3)

            # Upsample time back to T.
            if scale > 1:
                x_s = x_s.permute(0, 2, 3, 4, 1).reshape(B * V * J, d, t_cur)
                x_s = F.interpolate(x_s, size=T, mode="linear", align_corners=False)
                x_s = x_s.view(B, V, J, d, T).permute(0, 4, 1, 2, 3)

            scale_features.append(x_s)

        # Concatenate and fuse.
        x_cat = torch.cat(scale_features, dim=-1)  # (B, T, V, J, d * S)
        x_out = self.fusion(x_cat)
        return self.norm(x_out + x_in)


class _CameraConditioning(nn.Module):
    """Embed calibrated camera parameters and add them to per-view tokens.

    Parameters
    ----------
    d:
        Output per-view feature dimension.
    cam_dim:
        Hidden dimension of the camera encoder.
    n_views:
        Number of camera views (used to reshape camera tensors).
    """

    def __init__(self, d: int, cam_dim: int = 32, n_views: int = 4):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.encoder = nn.Sequential(
            nn.Linear(9 + 9 + 3, cam_dim),
            nn.ReLU(),
            nn.Linear(cam_dim, d),
        )

    def forward(
        self,
        feat: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Inject camera parameters into per-view tokens.

        Args
        ----
        feat:
            ``(N, V, J, d)`` per-view per-joint features (``J`` may be ``1``).
        K, R, t:
            Camera intrinsics / extrinsics with shapes ``(N, V, 3, 3)``,
            ``(N, V, 3, 3)``, ``(N, V, 3)``.

        Returns
        -------
        Conditioned features with the same shape as ``feat``.
        """
        N, V, J, d = feat.shape
        # Flatten joints to reuse a standard per-view encoder.
        feat_flat = feat.view(N * J, V, d)  # (N*J, V, d)
        # Repeat camera params for each joint.
        K_rep = K.unsqueeze(2).expand(-1, -1, J, -1, -1).reshape(N * J, V, 3, 3)
        R_rep = R.unsqueeze(2).expand(-1, -1, J, -1, -1).reshape(N * J, V, 3, 3)
        t_rep = t.unsqueeze(2).expand(-1, -1, J, -1).reshape(N * J, V, 3)
        cam_feat = torch.cat(
            [
                K_rep.view(N * J, V, -1),
                R_rep.view(N * J, V, -1),
                t_rep.view(N * J, V, -1),
            ],
            dim=-1,
        )
        conditioned = feat_flat + self.encoder(cam_feat)  # (N*J, V, d)
        return conditioned.view(N, V, J, d)


class _CameraConditionedEpipolarBias(nn.Module):
    """Compute per-frame epipolar attention bias from calibrated cameras.

    The bias is added to the cross-view attention scores inside the
    time+view transformer, encouraging the model to attend to geometrically
    consistent view pairs.
    """

    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        points_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-frame epipolar bias ``(N, V, V)``.

        Args
        ----
        K, R, t:
            ``(N, V, 3, 3)`` / ``(N, V, 3)`` camera parameters.
        points_2d:
            ``(N, V, J, 2)`` image points.

        Returns
        -------
        Per-frame bias ``(N, V, V)``.
        """
        return compute_per_frame_epipolar_bias(
            K, R, t, points_2d, temperature=self.temperature
        )


class OmniMultiViewFusionV3(OmniMultiViewFusionV2):
    """OmniMultiViewFusion v3 prototype.

    Parameters
    ----------
    use_multiscale_fusion:
        Enable the hierarchical multi-scale temporal/cross-view fusion block.
    use_camera_conditioning:
        Add camera-parameter embeddings to per-view tokens.
    use_epipolar_bias:
        Replace the spatio-temporal transformer with epipolar-biased layers.
    multiscale_scales:
        Temporal / joint downsample factors for the multi-scale block.
    camera_condition_dim:
        Hidden dimension of the camera encoder.
    epipolar_temperature:
        Temperature for the epipolar distance -> attention bias mapping.
    See ``OmniMultiViewFusionV2`` for the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 0,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_covariance: bool = True,
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
        graph_num_layers: int = 1,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        graph_dropout: float = 0.0,
        use_multiscale_fusion: bool = True,
        use_camera_conditioning: bool = True,
        use_epipolar_bias: bool = True,
        multiscale_scales: Sequence[int] = (1, 2, 4),
        camera_condition_dim: int = 32,
        epipolar_temperature: float = 10.0,
    ):
        # Initialise v2 ancestor.  Keep the dense joint-attention logic here too.
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_covariance=return_covariance,
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
            graph_num_layers=graph_num_layers,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
            graph_dropout=graph_dropout,
        )

        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_camera_conditioning = use_camera_conditioning
        self.use_epipolar_bias = use_epipolar_bias
        self.camera_condition_dim = camera_condition_dim
        self.epipolar_temperature = epipolar_temperature

        if self.use_multiscale_fusion:
            self.multiscale_fusion = _HierarchicalMultiscaleFusion(
                d=d,
                n_views=n_views,
                scales=multiscale_scales,
                n_heads=n_heads,
                dropout=0.1,
            )
        else:
            self.multiscale_fusion = None

        if self.use_camera_conditioning:
            self.camera_conditioning = _CameraConditioning(
                d=d, cam_dim=camera_condition_dim, n_views=n_views
            )
        else:
            self.camera_conditioning = None

        # Optionally replace the flat ST transformer with epipolar-biased layers.
        if self.use_epipolar_bias:
            self.st_transformer = nn.ModuleList(
                [
                    EpipolarBiasedTransformerEncoderLayer(
                        d_model=d,
                        nhead=n_heads,
                        dim_feedforward=d * 4,
                        dropout=0.1,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_st_layers)
                ]
            )
            self.epipolar_bias = _CameraConditionedEpipolarBias(
                temperature=epipolar_temperature
            )
        else:
            self.epipolar_bias = None

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[object] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import (
                _cameras_to_tensors,
            )
            K, R, t = _cameras_to_tensors(cameras, device)

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
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Optional dense joint-level self-attention (per-view) for capacity.
        if self.omni_joint_attn is not None:
            feat_j = feat.permute(0, 2, 1, 3).reshape(B * T * V, J, self.d)
            for layer in self.omni_joint_attn:
                feat_j = layer(feat_j)
            feat = feat_j.view(B * T, V, J, self.d)

        # Graph-joint attention over (view, joint) skeleton graph.
        feat = self._apply_graph_joint_attention(feat, J)  # (B*T, V, J, d)

        # Camera conditioning before multi-scale / ST fusion.
        if self.camera_conditioning is not None:
            feat = self.camera_conditioning(feat, K_corrected, R, t)

        # ---- NEW: hierarchical multi-scale temporal/cross-view fusion ----
        if self.multiscale_fusion is not None:
            feat = feat.view(B, T, V, J, self.d)
            feat = self.multiscale_fusion(feat)  # (B, T, V, J, d)
            feat = feat.view(B * T, V, J, self.d)

        # Spatio-temporal (time + view) attention with optional epipolar bias.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)

        # Compute epipolar attention bias if needed.
        if self.use_epipolar_bias:
            # points_2d: (B*T, V, J, 2); need per-frame bias (B*T, V, V).
            epi_bias = self.epipolar_bias(K_corrected, R, t, points_2d)  # (B*T, V, V)
            epi_bias = epi_bias.view(B, T, V, V)  # (B, T, V, V)
            attn_mask = build_temporal_bias_from_frames(
                epi_bias, n_heads=self.n_heads, n_joints=J
            )  # (B*J*n_heads, T*V, T*V)
            for layer in self.st_transformer:
                feat = layer(feat, epipolar_bias=attn_mask)
        else:
            for layer in self.st_transformer:
                feat = layer(feat)

        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(
            B * T, V, J, self.d
        )

        # Anisotropic covariance prediction per (view, joint).
        raw_cov = self.covariance_head(feat)  # (B*T, V, J, 3)
        L = self._cholesky_to_covariance(raw_cov)  # (B*T, V, J, 2, 2)

        # Precision weight = 1 / sqrt(det(Sigma)) = 1 / (l_xx * l_yy).
        precision = 1.0 / (
            L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
        )

        # Explicit visibility gating.
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * precision * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4, max=1e4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt

        # Fully batched DLT.
        from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

        pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        # Adaptive Gauss-Newton refinement.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        damping = self.damping_head(feat_pooled).squeeze(-1)  # (B*T, J)
        damping = self.min_gn_damping + (
            self.max_gn_damping - self.min_gn_damping
        ) * damping

        from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
            _adaptive_gauss_newton,
        )

        pred_3d_gn = _adaptive_gauss_newton(
            points_2d,
            weights,
            K_corrected,
            R,
            t,
            pred_3d_raw,
            damping,
            num_iters=self.gn_iters,
        )

        # Residual refinement head.
        residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_gn + delta

        # Epipolar consistency loss.
        epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
        epi_loss = self.epipolar_loss_weight * epi_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)
        visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)
            visibility = visibility.squeeze(1)

        # Build output tuple.
        out = (pred_3d, weights, visibility)
        if self.return_covariance:
            out += (L,)
        out += (epi_loss,)

        if self.return_pp_delta:
            out += (pp_delta,)
            if self.correct_focal:
                out += (focal_scale,)

        return out


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for smoke tests)."""
    from motionflow_mv.calibration.camera import Camera

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
    # CPU smoke test: shape/gradient sanity + single-frame compatibility.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = OmniMultiViewFusionV3(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
    )
    pred, weights, visibility, covariance, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)
    assert covariance.shape == (B, T, V, J, 2, 2)
    assert epi_loss.numel() == 1

    loss = pred.mean() + epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("OmniMultiViewFusionV3 multi-frame sanity check passed")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, weights4, visibility4, covariance4, epi4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert weights4.shape == (B, V, J)
    assert visibility4.shape == (B, V, J)
    assert covariance4.shape == (B, V, J, 2, 2)
    print("OmniMultiViewFusionV3 single-frame sanity check passed")

    # Ablation: new components disabled should still run.
    model_baseline = OmniMultiViewFusionV3(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=False,
        use_camera_conditioning=False,
        use_epipolar_bias=False,
    )
    out_baseline = model_baseline(x, cameras=cameras)
    assert out_baseline[0].shape == (B, T, J, 3)
    print("OmniMultiViewFusionV3 baseline ablation sanity check passed")
