"""OmniMultiViewFusion v4 — variable-view adaptive multi-view fusion.

OmniMultiViewFusionV4 subclasses :class:`OmniMultiViewFusionV3` and adds a set
of optional, independently togglable modules designed to close the variable-view
(2–4 views) robustness gap.  Every new component can be disabled so that v2/v3
checkpoints load with ``strict=False`` and the model falls back to the v3
behaviour.

New toggles
-----------
* ``use_context_visibility`` – context-aware per-view visibility gating.
* ``use_skeleton_residual`` – replace the dense residual MLP with a skeleton-
  graph residual refiner.
* ``use_kinematic_refiner`` – final kinematic-chain graph refiner on the output
  3-D pose.
* ``use_adaptive_view_selection`` – Gumbel-softmax adaptive view selector that
  learns to drop views per joint.
* ``use_rotation_correction`` – bounded SO(3) residual correction per view.
* ``use_entropy_regularization`` – attention-entropy regularisation on the
  per-view triangulation weights.

Input / output semantics follow v3:
    ``x``: (B, T, V, J, 3) or (B, V, J, 3) of (x_pixel, y_pixel, confidence)
    ``cameras`` or ``(K, R, t)``: calibrated camera parameters
Output (tuple):
    ``(pred_3d, weights, visibility, covariance, epipolar_loss, [extras])``
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from motionflow_mv.fusion.adaptive_view_selector import AdaptiveViewSelector
from motionflow_mv.fusion.attention_entropy_loss import AttentionEntropyLoss
from motionflow_mv.fusion.kinematic_chain_graph_refiner import (
    KinematicChainGraphRefiner,
)
from motionflow_mv.fusion.omniview_fusion_v3 import OmniMultiViewFusionV3
from motionflow_mv.fusion.rotation_correction import RotationCorrectionHead
from motionflow_mv.fusion.skeleton_graph_residual_refiner import (
    SkeletonGraphResidualRefiner,
)
from motionflow_mv.fusion.skeleton_graph_residual_refiner_v31 import (
    SkeletonGraphResidualRefinerV31,
)
from motionflow_mv.fusion.visibility_gated_fusion_v2 import (
    VisibilityGatedFusionV2,
)


class OmniMultiViewFusionV4(OmniMultiViewFusionV3):
    """OmniMultiViewFusion v4 prototype.

    Parameters
    ----------
    use_context_visibility:
        Use a context-aware visibility head.
    use_skeleton_residual:
        Replace the dense residual MLP with a skeleton-graph residual refiner.
    use_kinematic_refiner:
        Apply a final kinematic-chain graph refiner to the output pose.
    use_adaptive_view_selection:
        Use the Gumbel-softmax adaptive view selector and add a budget loss.
    use_rotation_correction:
        Predict a bounded per-view SO(3) residual and apply it before triangulation.
    use_entropy_regularization:
        Add an entropy regularisation term on the triangulation weights.
    adaptive_view_target_k:
        Target number of active views for the adaptive selector.
    rotation_max_rot_deg:
        Bound for the rotation correction residual (degrees).
    entropy_weight:
        Weight for the attention-entropy regularisation loss.
    See ``OmniMultiViewFusionV3`` for the remaining arguments.
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
        use_context_visibility: bool = False,
        use_skeleton_residual: bool = False,
        use_skeleton_residual_v31: bool = False,
        use_kinematic_refiner: bool = False,
        use_adaptive_view_selection: bool = False,
        use_rotation_correction: bool = False,
        use_entropy_regularization: bool = False,
        adaptive_view_target_k: int = 2,
        rotation_max_rot_deg: float = 2.0,
        entropy_weight: float = 0.01,
    ):
        # Keep v3 init exactly to preserve checkpoint compatibility.
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
            use_multiscale_fusion=use_multiscale_fusion,
            use_camera_conditioning=use_camera_conditioning,
            use_epipolar_bias=use_epipolar_bias,
            multiscale_scales=multiscale_scales,
            camera_condition_dim=camera_condition_dim,
            epipolar_temperature=epipolar_temperature,
        )

        self.use_context_visibility = use_context_visibility
        self.use_skeleton_residual = use_skeleton_residual
        self.use_skeleton_residual_v31 = use_skeleton_residual_v31
        self.use_kinematic_refiner = use_kinematic_refiner
        self.use_adaptive_view_selection = use_adaptive_view_selection
        self.use_rotation_correction = use_rotation_correction
        self.use_entropy_regularization = use_entropy_regularization

        # Optional context-aware visibility head.
        if self.use_context_visibility:
            self.context_visibility_gate = VisibilityGatedFusionV2(
                d=d,
                n_views=self.n_views,
                visibility_hidden=d // 2,
                visibility_threshold=visibility_threshold,
                min_visible_views=min_visible_views,
                use_context=True,
                use_uncertainty=False,
            )
        else:
            self.context_visibility_gate = None

        # Optional skeleton-graph residual refiner (drop-in for residual_mlp).
        if self.use_skeleton_residual_v31:
            self.residual_mlp = SkeletonGraphResidualRefinerV31(
                j=self.j,
                in_dim=self.d + 3,
                hidden_dim=residual_hidden,
                num_layers=2,
            )
        elif self.use_skeleton_residual:
            self.residual_mlp = SkeletonGraphResidualRefiner(
                j=self.j,
                in_dim=self.d + 3,
                hidden_dim=residual_hidden,
                num_layers=2,
            )

        # Optional final kinematic-chain refiner.
        if self.use_kinematic_refiner:
            self.kinematic_refiner = KinematicChainGraphRefiner(
                j=self.j, hidden_dim=d, num_layers=2
            )
        else:
            self.kinematic_refiner = None

        # Optional adaptive view selector.
        if self.use_adaptive_view_selection:
            self.adaptive_view_selector = AdaptiveViewSelector(
                d=d,
                n_views=self.n_views,
                n_joints=self.j,
                target_k=adaptive_view_target_k,
                budget_weight=0.01,
                use_selector=True,
            )
        else:
            self.adaptive_view_selector = None

        # Optional rotation correction head.
        if self.use_rotation_correction:
            self.rotation_correction_head = RotationCorrectionHead(
                d=d, hidden=d // 2, max_rot_deg=rotation_max_rot_deg
            )
        else:
            self.rotation_correction_head = None

        # Optional entropy regularisation.
        if self.use_entropy_regularization:
            self.attention_entropy_loss = AttentionEntropyLoss(
                weight=entropy_weight, dim=-2
            )
        else:
            self.attention_entropy_loss = None

    def _visibility_multiplier(
        self,
        feat: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-view/per-joint visibility multipliers in [0, 1]."""
        if self.use_context_visibility and self.context_visibility_gate is not None:
            return self.context_visibility_gate(feat, confidences)
        return super()._visibility_multiplier(feat, confidences)

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

        # Optional rotation correction on extrinsics.
        if self.use_rotation_correction and self.rotation_correction_head is not None:
            feat_rot = self._extract_frame_features(x_flat, K_corrected, R, t)
            feat_rot_pooled = feat_rot.mean(dim=2)  # (B*T, V, d)
            R, _ = self.rotation_correction_head(feat_rot_pooled, R)

        # Per-frame v3 features (uses corrected intrinsics and possibly corrected R).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)

        # Optional dense joint-level self-attention (per-view).
        if self.omni_joint_attn is not None:
            feat_j = feat.permute(0, 2, 1, 3).reshape(B * T * V, J, self.d)
            for layer in self.omni_joint_attn:
                feat_j = layer(feat_j)
            feat = feat_j.view(B * T, V, J, self.d)

        # Graph-joint attention over (view, joint) skeleton graph.
        feat = self._apply_graph_joint_attention(feat, J)

        # Camera conditioning.
        if self.camera_conditioning is not None:
            feat = self.camera_conditioning(feat, K_corrected, R, t)

        # Hierarchical multi-scale temporal/cross-view fusion.
        if self.multiscale_fusion is not None:
            feat = feat.view(B, T, V, J, self.d)
            feat = self.multiscale_fusion(feat)
            feat = feat.view(B * T, V, J, self.d)

        # Spatio-temporal (time + view) attention with optional epipolar bias.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)

        if self.use_epipolar_bias:
            from motionflow_mv.fusion.epipolar_transformer_bias import (
                build_temporal_bias_from_frames,
            )

            epi_bias = self.epipolar_bias(K_corrected, R, t, points_2d)
            epi_bias = epi_bias.view(B, T, V, V)
            attn_mask = build_temporal_bias_from_frames(
                epi_bias, n_heads=self.n_heads, n_joints=J
            )
            for layer in self.st_transformer:
                feat = layer(feat, epipolar_bias=attn_mask)
        else:
            for layer in self.st_transformer:
                feat = layer(feat)

        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(
            B * T, V, J, self.d
        )

        # Anisotropic covariance prediction per (view, joint).
        raw_cov = self.covariance_head(feat)
        L = self._cholesky_to_covariance(raw_cov)
        precision = 1.0 / (
            L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
        )

        # Visibility gating: optional context-aware head or v3 fallback.
        visibility = self._visibility_multiplier(feat, confidences)

        # Per-frame weight prediction and triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)

        # Optional adaptive view selection.
        budget_loss = torch.tensor(0.0, device=device)
        if self.use_adaptive_view_selection and self.adaptive_view_selector is not None:
            selector_mask, budget_loss = self.adaptive_view_selector(feat)
            weights = weights * selector_mask

        weights = weights * confidences * precision * visibility
        weights = weights.clamp(min=1e-4, max=1e4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K_corrected @ Rt

        from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

        pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        # Adaptive Gauss-Newton refinement.
        feat_pooled = feat.mean(dim=1)
        damping = self.damping_head(feat_pooled).squeeze(-1)
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
        residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d_gn + delta

        # Optional final kinematic-chain refiner.
        if self.use_kinematic_refiner and self.kinematic_refiner is not None:
            pred_3d = pred_3d + self.kinematic_refiner(pred_3d)

        # Epipolar consistency loss.
        epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
        epi_loss = self.epipolar_loss_weight * epi_loss

        # Optional entropy regularisation on triangulation weights.
        if (
            self.use_entropy_regularization
            and self.attention_entropy_loss is not None
        ):
            epi_loss = epi_loss + self.attention_entropy_loss(weights)

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)
        visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)
            visibility = visibility.squeeze(1)

        out = (pred_3d, weights, visibility, L, epi_loss)

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
    # T01 CPU smoke test: B=2, T=9, V=4, J=17.
    B, T, V, J = 2, 9, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    # Default toggle configuration (all new v4 toggles off, v3 path).
    model = OmniMultiViewFusionV4(
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
    print("OmniMultiViewFusionV4 default-toggle CPU smoke test passed (T=9)")

    # All v4 toggles enabled.
    model_full = OmniMultiViewFusionV4(
        j=J,
        d=64,
        n_views=V,
        graph_num_layers=1,
        use_multiscale_fusion=True,
        use_camera_conditioning=True,
        use_epipolar_bias=True,
        use_context_visibility=True,
        use_skeleton_residual=True,
        use_kinematic_refiner=True,
        use_adaptive_view_selection=True,
        use_rotation_correction=True,
        use_entropy_regularization=True,
    )
    pred2, weights2, visibility2, covariance2, epi_loss2 = model_full(x, cameras=cameras)
    assert pred2.shape == (B, T, J, 3)
    assert weights2.shape == (B, T, V, J)
    assert visibility2.shape == (B, T, V, J)
    assert covariance2.shape == (B, T, V, J, 2, 2)
    assert epi_loss2.numel() == 1
    loss2 = pred2.mean() + epi_loss2
    loss2.backward()
    assert any(p.grad is not None for p in model_full.parameters())
    print("OmniMultiViewFusionV4 full-toggle CPU smoke test passed (T=9)")
