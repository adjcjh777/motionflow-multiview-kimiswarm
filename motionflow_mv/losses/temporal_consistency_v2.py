"""Temporal velocity and acceleration consistency losses (v2).

This module extends the basic velocity + acceleration losses in
``motionflow_mv.losses.temporal_consistency`` with:

* optional Huber (smooth L1) robustness,
* per-joint weights / visibility masks,
* per-joint and per-frame loss outputs,
* safe handling of very short sequences (``T < 2`` for velocity,
  ``T < 3`` for acceleration).

The losses operate on the temporal dimension (default: axis ``-3``).
Input shapes are documented as ``(..., T, J, 3)`` for 3-D pose
sequences, but any shape with the temporal axis at ``dim=-3`` is
supported.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _finite_difference(
    x: torch.Tensor,
    order: int = 1,
    dim: int = -3,
) -> torch.Tensor:
    """Compute the ``order``-th finite difference of ``x`` along ``dim``.

    Args:
        x: Input tensor with temporal axis ``dim``.
        order: Order of the finite difference (1 for velocity, 2 for
            central acceleration).
        dim: Temporal axis.  Negative indexing is supported.

    Returns:
        ``n``-th order finite difference tensor.  The temporal size is
        reduced by ``order`` for forward differences.
    """
    if order == 1:
        return x[..., 1:, :, :] - x[..., :-1, :, :]
    if order == 2:
        return x[..., 2:, :, :] - 2.0 * x[..., 1:-1, :, :] + x[..., :-2, :, :]
    raise ValueError(f"Only order 1 and 2 are supported, got {order}")


def _apply_reduction(
    loss: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Apply mean/sum/no reduction to a raw per-element loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


def velocity_loss_v2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    joint_weights: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Temporal velocity consistency loss (v2).

    Compares first-order temporal finite differences between
    ``pred`` and ``gt``.  Optionally supports per-joint visibility
    masks, per-joint weighting, and Huber robustness.

    Args:
        pred: ``(..., T, J, 3)`` predicted 3-D joints.
        gt: ``(..., T, J, 3)`` ground-truth 3-D joints.
        mask: ``(..., T, J)`` boolean or float mask.  If provided,
            temporal frames where either endpoint is invalid are masked
            out.  ``True`` / ``1.0`` means valid.  Defaults to all
            valid.
        joint_weights: ``(J,)`` or ``(..., J)`` positive weights used
            to scale the per-joint contribution.  Defaults to uniform.
        loss_type: ``"l2"`` or ``"huber"``.
        delta: Huber threshold used when ``loss_type="huber"``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (or per-element loss if ``reduction="none"``).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    t_size = pred.shape[-3]
    if t_size < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_v = _finite_difference(pred, order=1)
    gt_v = _finite_difference(gt, order=1)
    diff = pred_v - gt_v  # (..., T-1, J, 3)

    if loss_type == "l2":
        per_frame = (diff ** 2).sum(dim=-1)  # (..., T-1, J)
    elif loss_type == "huber":
        norm = diff.norm(dim=-1)  # L2 norm of 3-D displacement
        quadratic = 0.5 * norm ** 2
        linear = delta * (norm - 0.5 * delta)
        per_frame = torch.where(norm <= delta, quadratic, linear)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if mask is not None:
        if mask.shape != pred.shape[:-1]:
            raise ValueError(
                f"mask shape {mask.shape} incompatible with pose shape {pred.shape[:-1]}"
            )
        valid_next = mask[..., 1:, :]
        valid_prev = mask[..., :-1, :]
        valid = (valid_next > 0).float() * (valid_prev > 0).float()
        per_frame = per_frame * valid

    if joint_weights is not None:
        # Broadcast over temporal and 3-D dimensions.
        while joint_weights.dim() < per_frame.dim():
            joint_weights = joint_weights.unsqueeze(0)
        per_frame = per_frame * joint_weights

    return _apply_reduction(per_frame, reduction)


def acceleration_loss_v2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    joint_weights: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Temporal acceleration consistency loss (v2).

    Compares second-order central finite differences between ``pred``
    and ``gt``.  Useful for penalising non-linear jitter while allowing
    uniform acceleration.

    Args:
        pred: ``(..., T, J, 3)`` predicted 3-D joints.
        gt: ``(..., T, J, 3)`` ground-truth 3-D joints.
        mask: ``(..., T, J)`` optional validity mask.
        joint_weights: ``(J,)`` or ``(..., J)`` optional per-joint weights.
        loss_type: ``"l2"`` or ``"huber"``.
        delta: Huber threshold used when ``loss_type="huber"``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (or per-element loss if ``reduction="none"``).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    t_size = pred.shape[-3]
    if t_size < 3:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_a = _finite_difference(pred, order=2)
    gt_a = _finite_difference(gt, order=2)
    diff = pred_a - gt_a

    if loss_type == "l2":
        per_frame = (diff ** 2).sum(dim=-1)
    elif loss_type == "huber":
        norm = diff.norm(dim=-1)  # L2 norm of 3-D displacement
        quadratic = 0.5 * norm ** 2
        linear = delta * (norm - 0.5 * delta)
        per_frame = torch.where(norm <= delta, quadratic, linear)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if mask is not None:
        if mask.shape != pred.shape[:-1]:
            raise ValueError(
                f"mask shape {mask.shape} incompatible with pose shape {pred.shape[:-1]}"
            )
        valid_mid = mask[..., 1:-1, :]
        valid_left = mask[..., :-2, :]
        valid_right = mask[..., 2:, :]
        valid = (valid_mid > 0).float() * (valid_left > 0).float() * (valid_right > 0).float()
        per_frame = per_frame * valid

    if joint_weights is not None:
        while joint_weights.dim() < per_frame.dim():
            joint_weights = joint_weights.unsqueeze(0)
        per_frame = per_frame * joint_weights

    return _apply_reduction(per_frame, reduction)


def temporal_consistency_loss_v2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    velocity_weight: float = 1.0,
    acceleration_weight: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    joint_weights: Optional[torch.Tensor] = None,
    loss_type: str = "l2",
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Combined temporal velocity + acceleration consistency loss (v2).

    Args:
        pred: ``(..., T, J, 3)`` predicted 3-D joints.
        gt: ``(..., T, J, 3)`` ground-truth 3-D joints.
        velocity_weight: Scalar weight for the first-order term.
        acceleration_weight: Scalar weight for the second-order term.
        mask: ``(..., T, J)`` optional validity mask.
        joint_weights: ``(J,)`` optional per-joint weights.
        loss_type: ``"l2"`` or ``"huber"``.
        delta: Huber threshold when ``loss_type="huber"``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Combined scalar loss (or per-element loss if ``reduction="none"``).
    """
    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    if velocity_weight != 0.0:
        total = total + velocity_weight * velocity_loss_v2(
            pred, gt, mask, joint_weights, loss_type, delta, reduction
        )
    if acceleration_weight != 0.0:
        total = total + acceleration_weight * acceleration_loss_v2(
            pred, gt, mask, joint_weights, loss_type, delta, reduction
        )
    return total


class TemporalConsistencyLossV2(nn.Module):
    """Configurable temporal velocity + acceleration loss.

    Parameters
    ----------
    velocity_weight:
        Weight for the first-order (velocity) consistency term.
    acceleration_weight:
        Weight for the second-order (acceleration) consistency term.
    loss_type:
        ``"l2"`` for squared error or ``"huber"`` for robust Huber loss.
    delta:
        Huber threshold when ``loss_type="huber"``.
    reduction:
        ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        loss_type: str = "l2",
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.loss_type = loss_type
        self.delta = delta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        joint_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the combined temporal consistency loss.

        Args:
            pred: ``(..., T, J, 3)`` predicted 3-D joints.
            gt: ``(..., T, J, 3)`` ground-truth 3-D joints.
            mask: ``(..., T, J)`` optional validity mask.
            joint_weights: ``(J,)`` optional per-joint weights.

        Returns:
            Scalar loss (or per-element loss if ``reduction="none"``).
        """
        return temporal_consistency_loss_v2(
            pred,
            gt,
            velocity_weight=self.velocity_weight,
            acceleration_weight=self.acceleration_weight,
            mask=mask,
            joint_weights=joint_weights,
            loss_type=self.loss_type,
            delta=self.delta,
            reduction=self.reduction,
        )
