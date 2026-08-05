"""Focal calibration loss for self-correcting camera intrinsics.

The model predicts a per-view focal-length scale ``s`` such that the corrected
intrinsic focal length is ``f_corrected = f_perturbed * s``.  During training the
intrinsics are perturbed by a known multiplicative factor ``p``.  The optimal
correction is therefore the inverse of that factor, ``s* = 1 / p``.  This loss
supervises the predicted scale to match that target.

Loss formula
------------

Given a batch of predicted scales :math:`s_{i,v}` and applied perturbation
scales :math:`p_{i,v}` (with :math:`p_{i,v} > 0`), the focal calibration loss
is a plain mean-squared error between the prediction and the ideal correction:

.. math::
    \mathcal{L}_{\text{focal}}
    = \frac{1}{N V} \sum_{i=1}^{N} \sum_{v=1}^{V}
        \left( s_{i,v} - \frac{1}{p_{i,v}} \right)^{2}.

A small constant ``eps`` is added to the denominator for numerical stability.
"""

import torch


def focal_calibration_loss(
    pred_scale: torch.Tensor,
    true_scale: torch.Tensor,
    weight: torch.Tensor = None,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the focal calibration (supervised focal-scale) loss.

    Args:
        pred_scale: Predicted focal-length correction scale, shape ``(..., V)``.
        true_scale: Applied focal-length perturbation scale, same shape as
            ``pred_scale``.  The perturbed focal length is ``f_perturbed = f_true * p``.
        weight: Optional per-view positive weights, same shape as ``pred_scale``.
        mask: Optional boolean mask (``True`` = valid), same shape as ``pred_scale``.
        eps: Small constant to avoid division by zero when inverting ``true_scale``.

    Returns:
        Scalar MSE loss tensor.
    """
    target = 1.0 / (true_scale.clamp(min=eps))
    sq = (pred_scale - target) ** 2

    if weight is not None:
        sq = sq * weight
    if mask is not None:
        sq = sq * mask.float()

    loss = sq.mean()
    return loss
