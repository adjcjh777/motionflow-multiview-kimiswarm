"""Reprojection loss for calibrated multi-view 3D pose.

Projects predicted world-coordinate 3D joints back into each calibrated view
and computes the 2D distance to the input keypoints.  Useful as an auxiliary
loss in addition to a 3D ground-truth MSE loss.
"""

import torch


def reprojection_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    confidences: torch.Tensor = None,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-view reprojection error.

    Args:
        pred_3d:   (B, T, J, 3) predicted world 3D joints.
        points_2d: (B, T, V, J, 2) input 2D keypoints (same unit as K).
        K:         (B, V, 3, 3) intrinsic matrices.
        R:         (B, V, 3, 3) rotation (world -> camera).
        t:         (B, V, 3) translation (world -> camera).
        confidences: (B, T, V, J) optional per-observation weights.
        mask:      (B, T, V, J) optional boolean mask (True = valid).
        eps:       small constant for numerical stability.

    Returns:
        Scalar reprojection MSE.
    """
    B, T, J, _ = pred_3d.shape
    V = K.shape[1]

    # Expand camera matrices over the temporal dimension.
    K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B,T,V,3,3)
    R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
    t = t.unsqueeze(1).expand(-1, T, -1, -1)

    # pred_3d: (B, T, J, 3) -> (B, T, 1, J, 3)
    X = pred_3d.unsqueeze(2)
    # Transform to camera space: X_cam = R @ X + t
    X_cam = torch.einsum("btvij,btvjk->btvik", R, X) + t.unsqueeze(-2)  # (B,T,V,J,3)

    # Project: x = K @ X_cam / z
    z = X_cam[..., 2:3]  # (B,T,V,J,1)
    proj = torch.einsum("btvij,btvjk->btvik", K, X_cam)  # (B,T,V,J,3)
    proj_2d = proj[..., :2] / (z.clamp(min=eps))

    diff = proj_2d - points_2d  # (B,T,V,J,2)
    sq = (diff ** 2).sum(dim=-1)  # (B,T,V,J)

    weight = torch.ones_like(sq)
    if confidences is not None:
        weight = weight * confidences
    if mask is not None:
        weight = weight * mask.float()

    if weight is not None:
        loss = (sq * weight).sum() / (weight.sum() + eps)
    else:
        loss = sq.mean()
    return loss
