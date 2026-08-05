"""Cross-view temporal residual + principal-point correction with explicit visibility gating.

Subclasses the best PP model and adds a small per-view/per-joint visibility head.
Visibility multiplies the DLT triangulation weights; a fallback guard ensures at
least ``min_visible_views`` views remain active per joint.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual model with learned visibility gating.

    Parameters
    ----------
    visibility_hidden:
        Hidden dimension of the visibility MLP (default 64).
    visibility_threshold:
        Threshold for the fallback guard (not used during training, only for
        diagnostics).
    min_visible_views:
        Minimum number of views to keep when applying the learned visibility.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining args.
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
        self.min_visible_views = min_visible_views
        self.visibility_head = nn.Sequential(
            nn.Linear(d, visibility_hidden),
            nn.ReLU(),
            nn.Linear(visibility_hidden, 1),
        )

    def _visibility_multiplier(self, feat, confidences):
        """Predict per-view/per-joint visibility and return a soft multiplier."""
        # feat: (B*T, V, J, d); confidences: (B*T, V, J)
        v_logits = self.visibility_head(feat).squeeze(-1)  # (B*T, V, J)
        visibility = torch.sigmoid(v_logits)

        # Fallback guard: if fewer than min_visible_views are predicted visible,
        # treat all views as visible for that joint.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (B*T, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)  # (B*T, 1, J)
        effective_visibility = visibility + (1.0 - visibility) * fallback

        # Also silence views that were already masked out by detector confidence.
        return effective_visibility * (confidences > 0).float()


if __name__ == "__main__":
    # Sanity check
    import torch
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility(
        j=28, d=64, n_views=14, n_st_layers=2, residual_hidden=128
    )
    x = torch.randn(1, 13, 14, 28, 3)
    K = torch.randn(14, 3, 3)
    K[:, 2, 2] = 1.0
    R = torch.eye(3).unsqueeze(0).expand(14, -1, -1)
    t = torch.randn(14, 3)
    pred, weights, visibility = model(x, K=K, R=R, t=t)
    assert pred.shape == (1, 13, 28, 3)
    assert weights.shape == (1, 13, 14, 28)
    assert visibility.shape == (1, 13, 14, 28)
    print("visibility-gated crossview PP model sanity check passed")
