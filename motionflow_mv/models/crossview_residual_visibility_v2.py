"""Visibility-aware adaptive fusion v2 for the cross-view residual PP model.

Extends ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` with
an explicit per-view, per-joint visibility head that gates the DLT
 triangulation weights.  The v2 visibility head is conditioned on a per-joint
pooled representation across views, so each view's visibility estimate is
aware of the full multi-view context.

Input / output semantics are identical to the parent class, except that the
forward pass always returns the effective visibility mask as the third tuple
item.
"""

import torch
import torch.nn as nn

from ..fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class CrossviewResidualVisibilityV2(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint):
    """Cross-view residual + principal-point model with visibility-gated DLT weights.

    Parameters
    ----------
    visibility_hidden:
        Hidden dimension of the visibility MLP (default 64).
    visibility_threshold:
        Probability threshold for the fallback guard.  Not used during training,
        only for the degenerate-view guard.
    min_visible_views:
        Minimum number of views to keep per joint when applying learned
        visibility.  If fewer views are predicted visible, all views are kept
        for that joint.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
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
        self.visibility_threshold = visibility_threshold
        self.min_visible_views = max(2, min_visible_views)

        # Visibility head: per-view feature + per-joint pooled context -> logit.
        self.visibility_head = nn.Sequential(
            nn.Linear(d * 2, visibility_hidden),
            nn.ReLU(),
            nn.Linear(visibility_hidden, 1),
        )

    def _visibility_multiplier(self, feat, confidences):
        """Predict per-view/per-joint visibility and return a soft multiplier.

        Args:
            feat: (N, V, J, d) spatio-temporal per-view features.
            confidences: (N, V, J) detector confidence scores.

        Returns:
            effective_visibility: (N, V, J) soft visibility multiplier in [0, 1].
        """
        # Per-joint pooled context across views.
        pooled = feat.mean(dim=1)  # (N, J, d)
        pooled = pooled.unsqueeze(1).expand_as(feat)  # (N, V, J, d)
        context = torch.cat([feat, pooled], dim=-1)  # (N, V, J, 2d)

        v_logits = self.visibility_head(context).squeeze(-1)  # (N, V, J)
        visibility = torch.sigmoid(v_logits)

        # Fallback guard: if fewer than min_visible_views are predicted visible,
        # treat all views as visible for that joint to avoid degenerate DLT.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (N, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)  # (N, 1, J)
        effective_visibility = visibility + (1.0 - visibility) * fallback

        # Silence views that were already masked out by detector confidence.
        return effective_visibility * (confidences > 0).float()
