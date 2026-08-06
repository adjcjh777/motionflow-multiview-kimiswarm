"""Robust reprojection-consistency losses for calibrated multi-view 3D pose.

These losses project predicted world-coordinate 3D joints back into each view
using the *corrected* intrinsics and compare the result with the observed 2D
keypoints.  They differ from ``reprojection_loss`` in ``reprojection.py`` by:

* returning raw per-joint/per-view error tensors (useful for masking/diagnostics);
* supporting robust norms (Charbonnier / Huber) to reduce outlier sensitivity;
* accepting explicit validity masks for occluded/dropped joints.
"""

import torch
import torch.nn.functional as F


def reprojection_error(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    confidences: torch.Tensor = None,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
):
    """Compute per-sample, per-view, per-joint reprojection error.

    Args
    ----
    pred_3d:   (B, T, J, 3) predicted world 3D joints.
    points_2d: (B, T, V, J, 2) input 2D keypoints.
    K:         (B, V, 3, 3) intrinsic matrices.
    R:         (B, V, 3, 3) rotation (world -> camera).
    t:         (B, V, 3) translation.
    confidences: (B, T, V, J) optional observation weights in [0, 1].
    mask:      (B, T, V, J) optional boolean validity mask (True = valid).
    eps:       small constant.

    Returns
    -------
    error: (B, T, V, J) per-joint reprojection error in pixels.
    """
    # Expand cameras over the temporal dimension.
    K = K.unsqueeze(1)  # (B, 1, V, 3, 3)
    R = R.unsqueeze(1)
    t = t.unsqueeze(1)

    # pred_3d: (B, T, J, 3) -> (B, T, 1, J, 3, 1)
    X = pred_3d.unsqueeze(2).unsqueeze(-1)
    # R: (B, 1, V, 3, 3) -> (B, 1, V, 1, 3, 3)
    R = R.unsqueeze(3)

    X_cam = (R @ X).squeeze(-1) + t.unsqueeze(-2)  # (B, T, V, J, 3)
    z = X_cam[..., 2:3]  # (B, T, V, J, 1)
    proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    proj_2d = proj[..., :2] / (z.clamp(min=eps))

    diff = proj_2d - points_2d  # (B, T, V, J, 2)
    error = (diff ** 2).sum(dim=-1).sqrt()  # (B, T, V, J)

    if confidences is not None:
        error = error * confidences
    if mask is not None:
        error = error * mask.float()

    return error


def robust_reprojection_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    confidences: torch.Tensor = None,
    mask: torch.Tensor = None,
    loss_type: str = "charbonnier",
    alpha: float = 1.0,
    eps: float = 1e-6,
    max_clip: float = None,
) -> torch.Tensor:
    """Robust reprojection loss.

    Args
    ----
    pred_3d, points_2d, K, R, t: as in ``reprojection_error``.
    confidences: optional (B, T, V, J) observation weights.
    mask: optional (B, T, V, J) boolean validity mask.
    loss_type: "charbonnier", "huber", or "mse".
    alpha: scale for Charbonnier / transition for Huber (pixels).
    eps: small constant for Charbonnier.
    max_clip: optional upper clip on per-joint error (pixels).

    Returns
    -------
    Scalar loss.
    """
    error = reprojection_error(pred_3d, points_2d, K, R, t, confidences=None, mask=None, eps=eps)

    if max_clip is not None:
        error = error.clamp(max=max_clip)

    if loss_type == "charbonnier":
        loss = (error ** 2 + alpha ** 2).sqrt() - alpha
    elif loss_type == "huber":
        loss = torch.where(error <= alpha, 0.5 * error ** 2 / alpha, error - 0.5 * alpha)
    elif loss_type == "mse":
        loss = error ** 2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    weight = torch.ones_like(loss)
    if confidences is not None:
        weight = weight * confidences
    if mask is not None:
        weight = weight * mask.float()

    denom = weight.sum() + eps
    return (loss * weight).sum() / denom
