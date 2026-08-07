"""Bayesian Tri v2 with learned per-view/per-joint visibility gating.

Subclasses ``RayAttentionFusionModelBayesianTriV2`` and replaces the default
``_visibility_multiplier`` (which returns 1) with a small MLP that predicts a
soft visibility mask. The predicted mask is returned as an extra output so the
trainer can add a BCE loss.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)


class RayAttentionFusionModelBayesianTriV2Visibility(
    RayAttentionFusionModelBayesianTriV2
):
    """Bayesian triangulation v2 with learned visibility gating.

    Parameters
    ----------
    visibility_hidden:
        Hidden dimension of the visibility MLP.
    visibility_threshold:
        Fallback threshold for the visibility guard.
    min_visible_views:
        Minimum number of views to keep per joint when visibility is applied.
    See ``RayAttentionFusionModelBayesianTriV2`` for the remaining args.
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
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
        return_covariance: bool = False,
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
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
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
            return_covariance=return_covariance,
        )
        self.visibility_threshold = visibility_threshold
        self.min_visible_views = min_visible_views
        self.visibility_head = nn.Sequential(
            nn.Linear(d, visibility_hidden),
            nn.ReLU(),
            nn.Linear(visibility_hidden, 1),
        )
        self._last_visibility = None

    def _visibility_multiplier(self, feat, confidences):
        """Predict per-view/per-joint visibility and return a soft multiplier."""
        # feat: (B*T, V, J, d); confidences: (B*T, V, J)
        v_logits = self.visibility_head(feat).squeeze(-1)  # (B*T, V, J)
        visibility = torch.sigmoid(v_logits)

        # Fallback guard: if fewer than min_visible_views are predicted visible,
        # treat all views as visible for that joint.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (B*T, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)
        effective_visibility = visibility + (1.0 - visibility) * fallback

        # Silence views that were already masked out by detector confidence.
        effective_visibility = effective_visibility * (confidences > 0).float()

        self._last_visibility = effective_visibility
        return effective_visibility

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape

        out = super().forward(x, cameras=cameras, K=K, R=R, t=t)

        # Remove the epipolar loss from the end, insert visibility, then
        # re-append the epipolar loss so outputs[-1] is always the auxiliary
        # loss for both bayesian_tri_v2 and bayesian_tri_v2_visibility.
        epi_loss = out[-1]
        out = out[:-1]

        # Build visibility tensor matching the final output shape.
        visibility = self._last_visibility
        if visibility is None:
            # Should not happen; provide a safe fallback.
            visibility = torch.ones((B * T, V, J), device=x.device, dtype=x.dtype)
        visibility = visibility.view(B, T, V, J)
        if squeeze_output:
            visibility = visibility.squeeze(1)

        return out + (visibility, epi_loss)
