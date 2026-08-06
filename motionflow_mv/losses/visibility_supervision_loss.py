"""Supervised visibility loss for occlusion-robust training.

During training we randomly occlude views/joints and supervise the predicted
visibility logits so that the model learns to recognise when a detection is
unreliable and should be down-weighted in triangulation.
"""

import torch
import torch.nn.functional as F


def visibility_supervision_loss(
    pred_logits: torch.Tensor,
    visible_mask: torch.Tensor,
    confidences: torch.Tensor = None,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCE between predicted visibility logits and a ground-truth visible mask.

    Parameters
    ----------
    pred_logits:
        Predicted visibility logits, shape (N, V, J) or (B, T, V, J).
    visible_mask:
        Binary mask with 1 = visible and 0 = occluded, same shape as ``pred_logits``.
    confidences:
        Optional detector confidences in [0, 1], same shape as ``pred_logits``.
        Positions with confidence == 0 are ignored in the loss.
    pos_weight:
        Weight for the visible (positive) class to balance occluded vs. visible tokens.

    Returns
    -------
    loss:
        Scalar BCE loss.
    """
    if confidences is not None:
        # Ignore detections that were already missing before synthetic occlusion.
        valid = confidences > 0
        if not valid.any():
            return pred_logits.new_zeros(())
        weights = valid.float()
    else:
        weights = torch.ones_like(pred_logits)

    # BCE with logits; apply class weighting via pos_weight.
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, visible_mask.float(), reduction="none"
    )
    # Up-weight visible tokens if requested.
    if pos_weight != 1.0:
        loss = loss * (1.0 + (pos_weight - 1.0) * visible_mask.float())

    loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)
    return loss
