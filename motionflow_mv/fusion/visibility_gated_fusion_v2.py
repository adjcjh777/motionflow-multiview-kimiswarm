"""Standalone visibility-gated fusion module (v2).

This module provides a reusable, standalone visibility gating mechanism for
multi-view 3D human pose fusion.  It predicts per-view, per-joint visibility
from spatio-temporal per-view features and multi-view context, and fuses the
resulting soft mask with detector confidences to drive triangulation weights.

The v2 design differs from :mod:`visibility_gated_fusion` in three ways:

1. **Context-aware visibility head** - each view's visibility estimate is
   conditioned on a per-joint pooled representation across all views, so the
   model can exploit multi-view consistency when deciding whether a joint is
   occluded in a given view.
2. **Uncertainty-aware fallback** - an optional learned uncertainty term
   modulates the soft mask; when too few views are visible, a deterministic
   fallback guard restores all views to prevent degenerate DLT.
3. **Standalone composition** - the module is intentionally small and can be
   plugged into any fusion model that produces ``(B*T, V, J, d)`` features and
   ``(B*T, V, J)`` confidences.

The file also includes a drop-in mixin for the current best cross-view residual
+ principal-point model, and a ``FusionModule`` wrapper for the pipeline.

Input / output semantics
------------------------
``VisibilityGatedFusionV2`` operates on tensors of shape ``(N, V, J, d)`` where
``N = B * T`` is the flattened batch.  It returns a soft visibility multiplier of
shape ``(N, V, J)`` in ``[0, 1]``.

``VisibilityGatedCrossviewResidualPrincipalPointV2`` follows the same
contract as ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
but additionally returns the effective visibility mask.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..calibration.camera import Camera
from .fusion_module import FusionModule
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class VisibilityGateHead(nn.Module):
    """Context-aware per-view/per-joint visibility head.

    Parameters
    ----------
    d:
        Feature dimension.
    hidden:
        Hidden dimension of the visibility MLP.
    use_context:
        If True, concatenate each view's feature with the per-joint mean-pooled
        feature across views before predicting visibility.
    use_uncertainty:
        If True, output an extra uncertainty channel that scales the visibility
        mask, allowing the model to express "don't know".
    use_context_visibility:
        Alias for ``use_context`` kept for the v4 model API.  If provided, it
        overrides ``use_context``.
    """

    def __init__(
        self,
        d: int = 64,
        hidden: int = 64,
        use_context: bool = True,
        use_uncertainty: bool = False,
        use_context_visibility: bool | None = None,
    ):
        super().__init__()
        # ``use_context_visibility`` is the v4-facing alias for ``use_context``.
        if use_context_visibility is not None:
            use_context = use_context_visibility
        self.use_context = use_context
        self.use_uncertainty = use_uncertainty

        in_dim = d * 2 if use_context else d
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(hidden // 2, 1)
        if use_uncertainty:
            self.uncertainty_head = nn.Linear(hidden // 2, 1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict visibility logits (and optional uncertainty).

        Args:
            feat: (N, V, J, d) per-view features.

        Returns:
            v_logits: (N, V, J) raw visibility logits.
            uncertainty: (N, V, J) uncertainty in (0, 1) if enabled, else None.
        """
        if self.use_context:
            # Per-joint pooled context across views.
            pooled = feat.mean(dim=1, keepdim=True).expand_as(feat)  # (N, V, J, d)
            x = torch.cat([feat, pooled], dim=-1)  # (N, V, J, 2d)
        else:
            x = feat

        x = self.mlp(x)
        v_logits = self.logit_head(x).squeeze(-1)  # (N, V, J)

        uncertainty = None
        if self.use_uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_head(x).squeeze(-1))

        return v_logits, uncertainty


class VisibilityGatedFusionV2(nn.Module):
    """Standalone visibility gating module for multi-view fusion.

    The module predicts per-view, per-joint visibility from spatio-temporal
    features and fuses it with detector confidences.  A fallback guard ensures
    that at least ``min_visible_views`` remain active for each joint to avoid
    degenerate triangulation.

    Parameters
    ----------
    d:
        Feature dimension.
    n_views:
        Number of views (used only for shape validation / diagnostics).
    visibility_hidden:
        Hidden dimension of the visibility MLP.
    visibility_threshold:
        Probability threshold below which a view is considered occluded by the
        fallback guard.
    min_visible_views:
        Minimum number of views that must remain visible per joint.  If fewer
        are predicted visible, all views are kept for that joint.
    use_context:
        If True, condition visibility on the per-joint pooled context.
    use_uncertainty:
        If True, predict an uncertainty map and use it to scale the visibility.
    use_context_visibility:
        Alias for ``use_context`` kept for the v4 model API.  If provided, it
        overrides ``use_context``.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        use_context: bool = True,
        use_uncertainty: bool = False,
        use_context_visibility: bool | None = None,
    ):
        super().__init__()
        # ``use_context_visibility`` is the v4-facing alias for ``use_context``.
        if use_context_visibility is not None:
            use_context = use_context_visibility
        self.d = d
        self.n_views = n_views
        self.visibility_threshold = visibility_threshold
        self.min_visible_views = max(2, min_visible_views)

        self.visibility_head = VisibilityGateHead(
            d=d,
            hidden=visibility_hidden,
            use_context=use_context,
            use_uncertainty=use_uncertainty,
        )
        self.use_uncertainty = use_uncertainty

    def forward(
        self,
        feat: torch.Tensor,
        confidences: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute the effective visibility multiplier.

        Args:
            feat: (N, V, J, d) per-view features.
            confidences: (N, V, J) detector confidence scores.
            return_logits:
                If True, also return the raw visibility logits.

        Returns:
            effective_visibility: (N, V, J) soft visibility multiplier in [0, 1].
            v_logits (optional): (N, V, J) raw visibility logits.
        """
        v_logits, uncertainty = self.visibility_head(feat)
        visibility = torch.sigmoid(v_logits)

        if self.use_uncertainty and uncertainty is not None:
            # Down-weight visibility where the model is uncertain (uncertainty=1).
            visibility = visibility * (1.0 - uncertainty)

        # Fallback guard: avoid degenerate DLT when too few views are visible.
        # The straight-through estimator keeps the hard forward decision (restore
        # all views to visible) while allowing gradients to flow through the
        # visibility head even when the guard fires.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (N, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)  # (N, 1, J)
        effective_visibility = visibility + (1.0 - visibility).detach() * fallback

        # Silence views that were already masked out by detector confidence.
        confidence_mask = (confidences > 0).float()
        effective_visibility = effective_visibility * confidence_mask

        if return_logits:
            return effective_visibility, v_logits
        return effective_visibility


class VisibilityGatedCrossviewResidualPrincipalPointV2(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view residual + PP model with the standalone v2 visibility gate.

    This is a drop-in replacement for the visibility-aware variants that embed
    the logic inline.  It delegates visibility estimation to
    ``VisibilityGatedFusionV2`` and returns the visibility mask together with
    the 3D pose.
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
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        use_context: bool = True,
        use_uncertainty: bool = False,
        use_context_visibility: bool | None = None,
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
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=True,
        )
        self.visibility_gate = VisibilityGatedFusionV2(
            d=d,
            n_views=n_views,
            visibility_hidden=visibility_hidden,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
            use_context=use_context,
            use_uncertainty=use_uncertainty,
            use_context_visibility=use_context_visibility,
        )

    def _visibility_multiplier(self, feat, confidences):
        return self.visibility_gate(feat, confidences)


class VisibilityGatedFusionV2Module(FusionModule):
    """Pipeline-ready ``FusionModule`` wrapper around the v2 visibility gate.

    Wraps ``VisibilityGatedCrossviewResidualPrincipalPointV2`` so it can be used
    by the rest of the MotionFlow multi-view pipeline.  Outputs are always in
    meters.
    """

    name = "visibility_gated_v2"

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        checkpoint_path: str | None = None,
        input_scale: float = 1.0,
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        use_context: bool = True,
        use_uncertainty: bool = False,
        use_context_visibility: bool | None = None,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.model = VisibilityGatedCrossviewResidualPrincipalPointV2(
            j=j,
            d=d,
            n_views=n_views,
            visibility_hidden=visibility_hidden,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
            use_context=use_context,
            use_uncertainty=use_uncertainty,
            use_context_visibility=use_context_visibility,
        )
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        points_2d = np.asarray(points_2d, dtype=np.float32)
        confidences = np.asarray(confidences, np.float32)

        if points_2d.ndim == 3:
            points_2d = points_2d[None]
        if confidences.ndim == 2:
            confidences = confidences[None]

        # Normalise cameras to meters.
        if self.input_scale != 1.0:
            cameras = [
                Camera(
                    K=cam.K.copy(),
                    R=cam.R.copy(),
                    t=cam.t.copy() / self.input_scale,
                )
                for cam in cameras
            ]

        T, V, J, _ = points_2d.shape
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, *_ = self.model(x_tensor, cameras)  # (T, J, 3)
            pred = pred.cpu().numpy()
        return pred


def register_visibility_gated_fusion_v2_module() -> None:
    """Register the visibility-gated v2 fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(VisibilityGatedFusionV2Module())


def _make_cameras(n_views: int = 4) -> List[Camera]:
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    from ..calibration.camera import Camera

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
    # Shape / gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = VisibilityGatedCrossviewResidualPrincipalPointV2(j=J, d=64, n_views=V)
    pred, weights, visibility = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert visibility.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("visibility-gated v2 model sanity check passed")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, weights4, visibility4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert weights4.shape == (B, V, J)
    assert visibility4.shape == (B, V, J)
    print("visibility-gated v2 single-frame sanity check passed")
