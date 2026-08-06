"""Masked-view 2D completion loss for self-supervised multi-view pose.

The model predicts a per-view 2D keypoint for every joint.  This loss compares
those predictions with the original 2D observations, but only on the slots that
were masked out during training.  It can therefore be trained without any 3D
supervision, yet still enforces physical consistency between the fused 3D
skeleton and each camera view.
"""

import torch


def masked_view_completion_loss(
    pred_2d: torch.Tensor,
    target_2d: torch.Tensor,
    mask: torch.Tensor,
    confidences: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Completion loss evaluated only on masked (B, T, V, J) slots.

    Args
    ----
    pred_2d:
        (B, T, V, J, 2) predicted 2D keypoints, typically obtained by
        reprojecting the fused 3D pose and refining with a small completion head.
    target_2d:
        (B, T, V, J, 2) original observed 2D keypoints.
    mask:
        (B, T, V, J) boolean tensor with ``True`` for masked-out slots.
    confidences:
        Optional (B, T, V, J) observation confidences in [0, 1].  If provided,
        masked slots should already have zero confidence.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    loss: scalar tensor.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred_2d.device)

    diff = pred_2d - target_2d  # (B, T, V, J, 2)
    error = diff.norm(dim=-1)  # (B, T, V, J)

    weight = mask.float()
    if confidences is not None:
        weight = weight * confidences

    denom = weight.sum() + eps
    return (error * weight).sum() / denom


def masked_view_completion_error(
    pred_2d: torch.Tensor,
    target_2d: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample mean pixel error on masked slots (useful for logging).

    Returns
    -------
    errors: (B,) tensor of average masked pixel errors.
    """
    diff = pred_2d - target_2d
    error = diff.norm(dim=-1)  # (B, T, V, J)
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (error * mask.float()).sum(dim=(1, 2, 3)) / denom
