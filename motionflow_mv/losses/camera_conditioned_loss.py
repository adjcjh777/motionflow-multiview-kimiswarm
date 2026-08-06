"""Camera-parameter-conditioned auxiliary losses.

These losses explicitly use the calibrated camera rig to encourage both
reprojection consistency and a physically plausible skeleton scale.  They are
intended as optional drop-in auxiliaries for the camera-conditioned fusion
model.
"""

import torch
import torch.nn.functional as F


def camera_conditioned_reprojection_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor = None,
    mask: torch.Tensor = None,
    loss_type: str = "charbonnier",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Robust reprojection loss with optional per-view weighting.

    Parameters
    ----------
    pred_3d:
        Predicted 3D joints, ``(B, T, J, 3)``.
    points_2d:
        Input 2D keypoints, ``(B, T, V, J, 2)``.
    K, R, t:
        Camera intrinsics and extrinsics, ``(B, V, 3, 3)`` / ``(B, V, 3)``.
    weights:
        Per-view weights, ``(B, T, V, J)``.  If ``None``, uniform weighting is used.
    mask:
        Optional boolean mask, ``(B, T, V, J)`` (``True`` = valid).
    loss_type:
        ``"mse"`` or ``"charbonnier"``.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    Scalar reprojection loss.
    """
    # Expand cameras over the temporal dimension.
    K = K.unsqueeze(1)  # (B, 1, V, 3, 3)
    R = R.unsqueeze(1)
    t = t.unsqueeze(1)

    # Transform 3D points into each camera frame.
    X = pred_3d.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, J, 3, 1)
    R = R.unsqueeze(3)  # (B, 1, V, 1, 3, 3)
    X_cam = (R @ X).squeeze(-1) + t.unsqueeze(-2)  # (B, T, V, J, 3)

    # Project.
    z = X_cam[..., 2:3]
    proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)
    proj_2d = proj[..., :2] / (z.clamp(min=eps))

    diff = proj_2d - points_2d  # (B, T, V, J, 2)
    sq = (diff ** 2).sum(dim=-1)  # (B, T, V, J)

    if loss_type == "mse":
        err = sq
    elif loss_type == "charbonnier":
        err = torch.sqrt(sq + eps)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    weight = torch.ones_like(err)
    if weights is not None:
        weight = weight * weights
    if mask is not None:
        weight = weight * mask.float()

    return (err * weight).sum() / (weight.sum() + eps)


def camera_conditioned_scale_loss(
    pred_3d: torch.Tensor,
    parents: list[int],
    reduction: str = "mean",
) -> torch.Tensor:
    """Temporal bone-length consistency loss.

    Encourages the predicted 3D skeleton to maintain a stable bone length over
    time, which is a weak physical prior that is independent of any single
    camera view.

    Parameters
    ----------
    pred_3d:
        Predicted 3D joints, ``(B, T, J, 3)``.
    parents:
        Parent index for each joint; ``-1`` for roots.
    reduction:
        ``"mean"`` or ``"sum"``.

    Returns
    -------
    Scalar loss tensor.
    """
    if pred_3d.shape[-3] < 2:
        return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)

    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bones.append(pred_3d[..., child, :] - pred_3d[..., parent, :])

    if len(bones) == 0:
        return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)

    bones = torch.stack(bones, dim=-2)  # (B, T, B_n, 3)
    lengths = bones.norm(dim=-1)  # (B, T, B_n)

    # Penalize frame-to-frame bone-length changes.
    delta = lengths[:, 1:] - lengths[:, :-1]  # (B, T-1, B_n)
    loss = delta ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
