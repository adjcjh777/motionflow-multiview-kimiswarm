"""Training-time camera calibration perturbations.

Utilities to corrupt intrinsics and extrinsics with realistic noise so the
fusion model learns to be robust to small calibration errors.
"""

from __future__ import annotations

import torch


def so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3).

    Uses Rodrigues' rotation formula.
    """
    shape = axis_angle.shape[:-1]
    theta = axis_angle.norm(dim=-1, keepdim=True)[..., None]  # (..., 1, 1)
    k = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)

    # Cross-product matrix K
    K = torch.zeros(*shape, 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]

    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    I = I.view(*([1] * len(shape)), 3, 3).expand(*shape, 3, 3)

    sin = torch.sin(theta)
    cos = 1 - torch.cos(theta)
    return I + sin * K + cos * (K @ K)


def perturb_intrinsics(
    K: torch.Tensor,
    focal_std: float = 0.0,
    pp_std: float = 0.0,
) -> torch.Tensor:
    """Perturb intrinsic matrices.

    Args:
        K: Intrinsic matrices of shape (..., 3, 3).
        focal_std: Relative standard deviation of focal length (e.g. 0.01 = 1%).
        pp_std: Standard deviation of principal point in pixels.

    Returns:
        Perturbed intrinsics of the same shape.
    """
    K_out, _, _ = perturb_intrinsics_with_delta(K, focal_std=focal_std, pp_std=pp_std)
    return K_out


def perturb_intrinsics_with_delta(
    K: torch.Tensor,
    focal_std: float = 0.0,
    pp_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb intrinsic matrices and return the applied offsets.

    Args:
        K: Intrinsic matrices of shape (..., 3, 3).
        focal_std: Relative standard deviation of focal length (e.g. 0.01 = 1%).
        pp_std: Standard deviation of principal point in pixels.

    Returns:
        Tuple of (perturbed K, principal point offset in pixels, focal scale).
        The principal-point offset has shape (..., 2) and the focal scale has
        shape (..., 1).  focal_scale is the multiplicative factor applied to
        fx and fy, i.e. ``fx' = fx * focal_scale``.
    """
    K_out = K.clone()
    delta = torch.zeros(*K.shape[:-2], 2, device=K.device, dtype=K.dtype)
    focal_scale = torch.ones(*K.shape[:-2], 1, device=K.device, dtype=K.dtype)
    if focal_std > 0:
        noise = torch.randn(K.shape[:-2], device=K.device, dtype=K.dtype) * focal_std
        scale = 1 + noise
        K_out[..., 0, 0] *= scale
        K_out[..., 1, 1] *= scale
        focal_scale = scale.unsqueeze(-1)
    if pp_std > 0:
        noise = torch.randn(*K.shape[:-2], 2, device=K.device, dtype=K.dtype) * pp_std
        K_out[..., 0, 2] += noise[..., 0]
        K_out[..., 1, 2] += noise[..., 1]
        delta = noise
    return K_out, delta, focal_scale


def perturb_extrinsics(
    R: torch.Tensor,
    t: torch.Tensor,
    rot_std: float = 0.0,
    trans_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb extrinsic parameters.

    Args:
        R: Rotation matrices of shape (..., 3, 3).
        t: Translation vectors of shape (..., 3).
        rot_std: Rotation noise standard deviation in degrees.
        trans_std: Translation noise standard deviation in the same unit as t.

    Returns:
        Perturbed (R, t) of the same shapes.
    """
    if rot_std <= 0 and trans_std <= 0:
        return R, t
    R_out = R
    t_out = t
    if rot_std > 0:
        axis = torch.randn(*R.shape[:-2], 3, device=R.device, dtype=R.dtype)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        theta = torch.randn(*R.shape[:-2], 1, device=R.device, dtype=R.dtype) * (rot_std * 3.14159265 / 180.0)
        delta_R = so3_exp((axis * theta).squeeze(-1))
        R_out = torch.einsum("...ij,...jk->...ik", delta_R, R)
    if trans_std > 0:
        t_out = t + torch.randn_like(t) * trans_std
    return R_out, t_out


def perturb_cameras(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    *,
    rot_std: float = 0.0,
    trans_std: float = 0.0,
    focal_std: float = 0.0,
    pp_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply independent per-view perturbations to calibrated cameras.

    Args:
        K, R, t: Intrinsics and extrinsics.  Accepts both per-sequence
            (V, 3, 3) / (V, 3) and batched (B, V, 3, 3) / (B, V, 3) inputs.
        rot_std, trans_std, focal_std, pp_std: Standard deviations for each
            perturbation type.  Set to 0 to skip that perturbation.

    Returns:
        Tuple of perturbed (K, R, t).
    """
    K_aug, _ = perturb_intrinsics_with_delta(K, focal_std=focal_std, pp_std=pp_std)
    R_aug, t_aug = perturb_extrinsics(R, t, rot_std=rot_std, trans_std=trans_std)
    return K_aug, R_aug, t_aug


def perturb_cameras_with_delta(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    *,
    rot_std: float = 0.0,
    trans_std: float = 0.0,
    focal_std: float = 0.0,
    pp_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply independent per-view perturbations and return the applied offsets.

    Returns:
        Tuple of perturbed (K, R, t, pp_delta, focal_scale), where pp_delta is
        the per-view principal-point offset of shape (..., V, 2) in pixels and
        focal_scale is the per-view focal-length scale of shape (..., V, 1).
    """
    K_aug, pp_delta, focal_scale = perturb_intrinsics_with_delta(K, focal_std=focal_std, pp_std=pp_std)
    R_aug, t_aug = perturb_extrinsics(R, t, rot_std=rot_std, trans_std=trans_std)
    return K_aug, R_aug, t_aug, pp_delta, focal_scale
