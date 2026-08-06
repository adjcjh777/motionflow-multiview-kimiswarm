"""SSL-specific cross-view contrastive variant of the ray-attention anchor.

During self-supervised pretraining we want the cross-view contrastive loss to be
emitted from every forward pass so the training loop can mix it in with the
masked-view reprojection objective.  This thin wrapper subclasses the existing
supervised cross-view contrastive model and overrides ``forward`` to call
``forward_with_contrastive_loss``, returning ``(pred_3d, weights, c_loss)``.
"""

from .ray_attention_temporal_crossview_residual_principal_point_crossview_contrast_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSSLViewContrast(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast,
):
    """Cross-view contrastive ray-attention model wired for masked-view SSL pretraining.

    Parameters
    ----------
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast``.
    The contrastive loss weight defaults to ``1.0`` so the training script can
    apply its own ``lambda_contrast`` scaling without double-counting.
    """

    def __init__(self, *args, contrastive_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, contrastive_loss_weight=contrastive_loss_weight, **kwargs)

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        """Run the anchor forward and also return the cross-view contrastive loss.

        Returns
        -------
        pred_3d: (B, T, J, 3) fused 3D pose.
        weights: (B, T, V, J) per-view contribution weights.
        c_loss: scalar contrastive loss, already weighted by ``contrastive_loss_weight``.
        """
        return self.forward_with_contrastive_loss(x, cameras=cameras, K=K, R=R, t=t)
