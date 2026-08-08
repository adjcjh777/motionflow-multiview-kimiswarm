"""v26: differentiable camera calibration refinement sub-module.

Given an initial 3-D skeleton and calibrated views, this module refines the
per-camera intrinsics and/or extrinsics by taking a small number of
gradient-descent steps on the weighted reprojection loss. The 3-D skeleton is
kept fixed, so the module is a pure *camera* refinement block that can be
plugged into any pose-estimation pipeline.

The update is gated by a learnable scalar initialised to zero, which keeps the
module an identity mapping at the start of training and lets it open gradually.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .perturb import so3_exp


def _build_K(fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """Assemble pinhole intrinsics from per-element focal lengths and principal point.

    Args:
        fx, fy, cx, cy: tensors of shape (..., V).

    Returns:
        K: (..., V, 3, 3) intrinsics.
    """
    *leading, V = fx.shape
    device, dtype = fx.device, fx.dtype
    K = torch.zeros(*leading, V, 3, 3, device=device, dtype=dtype)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0
    return K


def _reprojection_error(
    X: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-point reprojection error and validity mask.

    Args:
        X: (B, T, J, 3).
        points_2d: (B, T, V, J, 2).
        K: (B, T, V, 3, 3).
        R: (B, T, V, 3, 3).
        t: (B, T, V, 3).
        weights: (B, T, V, J).

    Returns:
        diff: (B, T, V, J) reprojection L2 distance.
        valid: (B, T, V, J) bool mask of points with positive depth.
    """
    # X_cam = R @ X + t
    X_exp = X.unsqueeze(2)  # (B, T, 1, J, 3)
    X_cam = torch.matmul(R.unsqueeze(3), X_exp.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    X_cam = X_cam + t.unsqueeze(3)
    Z = X_cam[..., 2]
    valid = Z > 1e-4

    proj = torch.matmul(K.unsqueeze(3), X_cam.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    Z_safe = Z.sign() * (Z.abs() + 1e-6)
    uv = proj[..., :2] / Z_safe.unsqueeze(-1)
    diff = (uv - points_2d).norm(dim=-1)  # (B, T, V, J)
    return diff, valid


def _reprojection_loss(
    X: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    view_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean weighted reprojection loss, normalised by a canonical pixel scale."""
    B, T, V, J = points_2d.shape[:4]
    if weights is None:
        weights = torch.ones(B, T, V, J, device=points_2d.device, dtype=points_2d.dtype)

    diff, valid = _reprojection_error(X, points_2d, K, R, t, weights)
    w = weights * valid.float()
    if view_mask is not None:
        w = w * view_mask[:, :, :, None].float()

    loss = (diff * w).sum() / w.sum().clamp(min=1e-6)
    # Normalise by a canonical image scale (pixels) so loss is O(1).
    return loss / 1000.0


class CameraRefinementV26(nn.Module):
    """Differentiable camera calibration refinement.

    Parameters
    ----------
    n_steps:
        Number of gradient-descent steps on the camera parameters (default 2).
    lr:
        Step-size multiplier for each gradient-descent step (default 0.05).
    refine_intrinsics:
        Whether to refine focal length and principal point.
    refine_extrinsics:
        Whether to refine rotation and translation.
    max_focal_scale:
        Maximum multiplicative focal-length update (e.g. 0.05 = 5%%).
    max_principal_point_update:
        Maximum principal-point update in pixels.
    max_rotation_deg:
        Maximum rotation correction per step in degrees.
    max_translation:
        Maximum translation correction per step in metres.

    Notes
    -----
    The refined cameras are returned as a residual around the input cameras,
    scaled by a learnable gate initialised to zero. This makes the layer an
    identity mapping at the start of training and gives a stable warm start.
    """

    def __init__(
        self,
        n_steps: int = 2,
        lr: float = 0.05,
        refine_intrinsics: bool = True,
        refine_extrinsics: bool = True,
        max_focal_scale: float = 0.05,
        max_principal_point_update: float = 5.0,
        max_rotation_deg: float = 2.0,
        max_translation: float = 0.05,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.lr = lr
        self.refine_intrinsics = refine_intrinsics
        self.refine_extrinsics = refine_extrinsics
        self.max_focal_scale = max_focal_scale
        self.max_principal_point_update = max_principal_point_update
        self.max_rotation_rad = max_rotation_deg * 3.141592653589793 / 180.0
        self.max_translation = max_translation

        # Scalar gate, initialised to 0 => identity at init.
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _intrinsics_from_params(
        self,
        K_in: torch.Tensor,
        log_fx: torch.Tensor,
        log_fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
    ) -> torch.Tensor:
        """Build intrinsics from decomposed parameters, clamped to sane ranges."""
        fx = K_in[..., 0, 0] * torch.clamp(torch.exp(log_fx), 1.0 - self.max_focal_scale, 1.0 + self.max_focal_scale)
        fy = K_in[..., 1, 1] * torch.clamp(torch.exp(log_fy), 1.0 - self.max_focal_scale, 1.0 + self.max_focal_scale)
        cx = K_in[..., 0, 2] + torch.clamp(cx, -self.max_principal_point_update, self.max_principal_point_update)
        cy = K_in[..., 1, 2] + torch.clamp(cy, -self.max_principal_point_update, self.max_principal_point_update)
        return _build_K(fx, fy, cx, cy)

    def _rotation_from_params(self, R_in: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """Apply bounded axis-angle update to the input rotation matrices."""
        axis = torch.clamp(axis, -self.max_rotation_rad, self.max_rotation_rad)
        *leading, V, _ = axis.shape
        R_delta = so3_exp(axis.reshape(-1, 3)).reshape(*leading, V, 3, 3)
        return torch.matmul(R_delta, R_in)

    def _translation_from_params(self, t_in: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        delta = torch.clamp(delta, -self.max_translation, self.max_translation)
        return t_in + delta

    def forward(
        self,
        points_2d: torch.Tensor,
        X: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine camera parameters for the given fixed 3-D skeleton.

        Args:
            points_2d: (B, T, V, J, 2) observed image keypoints.
            X: (B, T, J, 3) fixed 3-D skeleton in world coordinates.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations (world -> camera).
            t: (B, T, V, 3) translations (world -> camera).
            weights: optional (B, T, V, J) per-keypoint weights.
            view_mask: optional (B, T, V) bool; True = valid view.

        Returns:
            K_ref: (B, T, V, 3, 3) refined intrinsics.
            R_ref: (B, T, V, 3, 3) refined rotations.
            t_ref: (B, T, V, 3) refined translation.
        """
        if view_mask is not None:
            view_mask = view_mask.bool()

        # Initialise parameter leaves around identity (zero) corrections.
        log_fx = torch.zeros_like(K[..., 0, 0]).detach().requires_grad_(self.refine_intrinsics)
        log_fy = torch.zeros_like(K[..., 1, 1]).detach().requires_grad_(self.refine_intrinsics)
        cx = torch.zeros_like(K[..., 0, 2]).detach().requires_grad_(self.refine_intrinsics)
        cy = torch.zeros_like(K[..., 1, 2]).detach().requires_grad_(self.refine_intrinsics)

        axis = torch.zeros_like(t).detach().requires_grad_(self.refine_extrinsics)
        delta_t = torch.zeros_like(t).detach().requires_grad_(self.refine_extrinsics)

        for _ in range(self.n_steps):
            K_step = self._intrinsics_from_params(K, log_fx, log_fy, cx, cy)
            R_step = self._rotation_from_params(R, axis)
            t_step = self._translation_from_params(t, delta_t)

            loss = _reprojection_loss(X, points_2d, K_step, R_step, t_step, weights, view_mask)

            grads = torch.autograd.grad(
                loss,
                [log_fx, log_fy, cx, cy, axis, delta_t],
                create_graph=True,
                allow_unused=True,
            )

            with torch.no_grad():
                if self.refine_intrinsics:
                    log_fx = log_fx - self.lr * grads[0]
                    log_fy = log_fy - self.lr * grads[1]
                    cx = cx - self.lr * grads[2]
                    cy = cy - self.lr * grads[3]
                if self.refine_extrinsics:
                    axis = axis - self.lr * grads[4]
                    delta_t = delta_t - self.lr * grads[5]

                # Re-attach for the next iteration.
                log_fx = log_fx.detach().requires_grad_(self.refine_intrinsics)
                log_fy = log_fy.detach().requires_grad_(self.refine_intrinsics)
                cx = cx.detach().requires_grad_(self.refine_intrinsics)
                cy = cy.detach().requires_grad_(self.refine_intrinsics)
                axis = axis.detach().requires_grad_(self.refine_extrinsics)
                delta_t = delta_t.detach().requires_grad_(self.refine_extrinsics)

        # Final refined parameters.
        K_step = self._intrinsics_from_params(K, log_fx, log_fy, cx, cy)
        R_step = self._rotation_from_params(R, axis)
        t_step = self._translation_from_params(t, delta_t)

        # Gated residual update keeps the module identity at init.
        gate = torch.tanh(self.residual_scale)
        K_ref = K + gate * (K_step - K)
        # Rotation residual: blend the axis-angle update.
        axis_residual = gate * axis
        R_ref = self._rotation_from_params(R, axis_residual)
        t_ref = t + gate * (t_step - t)
        return K_ref, R_ref, t_ref
