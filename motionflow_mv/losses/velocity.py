"""Temporal velocity consistency loss.

Penalizes differences between predicted and ground-truth first-order
velocities along the temporal axis.  Useful for encouraging smoother,
less jittery pose estimates when clips are longer than a single frame.
"""

import torch


def velocity_loss(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """L2 velocity loss between predicted and ground-truth 3D poses.

    Args:
        pred: (..., T, J, 3) predicted 3D joints.
        gt: (..., T, J, 3) ground-truth 3D joints.
        reduction: "mean" or "sum".

    Returns:
        Scalar velocity loss (mean or sum of squared velocity errors).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if pred.shape[-3] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_v = pred[..., 1:, :, :] - pred[..., :-1, :, :]
    gt_v = gt[..., 1:, :, :] - gt[..., :-1, :, :]
    diff = pred_v - gt_v
    loss = (diff ** 2).sum(dim=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def velocity_l1_loss(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """L1 velocity loss (Huber-like, less sensitive to single-frame outliers)."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if pred.shape[-3] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_v = pred[..., 1:, :, :] - pred[..., :-1, :, :]
    gt_v = gt[..., 1:, :, :] - gt[..., :-1, :, :]
    diff = pred_v - gt_v
    loss = diff.norm(dim=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
