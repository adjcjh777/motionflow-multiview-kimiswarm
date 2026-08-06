"""Temporal velocity and acceleration consistency losses.

Builds on ``motionflow_mv.losses.velocity`` by adding second-order finite
difference (acceleration) consistency.  The combined loss can be used as a
training auxiliary term for temporal sequences of 3-D joints.
"""

import torch
import torch.nn as nn


def acceleration_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L2 acceleration loss between predicted and ground-truth 3-D pose.

    Computes the second-order central finite difference along the temporal
    dimension and penalises the squared error.  For perfectly linear motion
    (constant velocity) the loss is zero when ``pred == gt``.

    Args:
        pred: (..., T, J, 3) predicted 3D joints.
        gt: (..., T, J, 3) ground-truth 3D joints.
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        Scalar acceleration loss.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if pred.shape[-3] < 3:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_a = pred[..., 2:, :, :] - 2.0 * pred[..., 1:-1, :, :] + pred[..., :-2, :, :]
    gt_a = gt[..., 2:, :, :] - 2.0 * gt[..., 1:-1, :, :] + gt[..., :-2, :, :]
    diff = pred_a - gt_a
    loss = (diff ** 2).sum(dim=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def temporal_consistency_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Combined temporal velocity + acceleration consistency loss.

    Args:
        pred: (..., T, J, 3) predicted 3D joints.
        gt: (..., T, J, 3) ground-truth 3D joints.
        velocity_weight: Scalar weight for the velocity term.
        acceleration_weight: Scalar weight for the acceleration term.
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        Scalar combined loss.
    """
    from .velocity import velocity_loss

    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    if velocity_weight != 0.0:
        total = total + velocity_weight * velocity_loss(pred, gt, reduction=reduction)
    if acceleration_weight != 0.0:
        total = total + acceleration_weight * acceleration_loss(pred, gt, reduction=reduction)
    return total


class TemporalConsistencyLoss(nn.Module):
    """Convenience module wrapper for temporal velocity + acceleration loss.

    Parameters
    ----------
    velocity_weight:
        Weight for the first-order (velocity) consistency term.
    acceleration_weight:
        Weight for the second-order (acceleration) consistency term.
    reduction:
        ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute the combined temporal consistency loss.

        Args:
            pred: (..., T, J, 3) predicted 3D joints.
            gt: (..., T, J, 3) ground-truth 3D joints.

        Returns:
            Scalar loss tensor.
        """
        return temporal_consistency_loss(
            pred,
            gt,
            velocity_weight=self.velocity_weight,
            acceleration_weight=self.acceleration_weight,
            reduction=self.reduction,
        )
