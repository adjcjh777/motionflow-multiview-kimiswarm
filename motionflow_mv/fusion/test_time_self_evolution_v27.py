"""Test-time self-evolution via iterative geometric self-consistency.

This module implements a lightweight, gradient-free inference loop that
refines a 3D pose estimate by re-weighting views according to their
reprojection residuals.  It is meant to be inserted after the v25/v26
geometry-fusion block and used only at ``model.eval()``.

The loop is intentionally simple: it does **not** update any model parameters;
only the per-sample view confidences evolve.  This matches the "self-evolution"
idea in the Qwen3.8 blog: the system refines its own output by checking it
against the input evidence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.triangulation import triangulate_dlt_torch


def build_projection_matrix(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build (B, T, V, 3, 4) projection matrices P = K [R | t].

    Args:
        K: (B, T, V, 3, 3) intrinsics.
        R: (B, T, V, 3, 3) rotations.
        t: (B, T, V, 3) translations.

    Returns:
        P: (B, T, V, 3, 4) projection matrices.
    """
    B, T, V, _, _ = K.shape
    # K @ R -> (B, T, V, 3, 3)
    KR = torch.matmul(K, R)
    # K @ t -> (B, T, V, 3, 1)
    Kt = torch.matmul(K, t.unsqueeze(-1))
    P = torch.cat([KR, Kt], dim=-1)  # (B, T, V, 3, 4)
    return P


def compute_reprojection_residual(
    X: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute per-view per-joint L2 reprojection error in pixels.

    Args:
        X: (B, T, J, 3) 3D joint positions.
        points_2d: (B, T, V, J, 2) detected 2D keypoints.
        K: (B, T, V, 3, 3) intrinsics.
        R: (B, T, V, 3, 3) rotations.
        t: (B, T, V, 3) translations.

    Returns:
        residual: (B, T, V, J) L2 reprojection error.
    """
    X_exp = X.unsqueeze(2).expand(-1, -1, K.shape[2], -1, -1)  # (B, T, V, J, 3)
    X_exp = X_exp.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
    X_cam = torch.matmul(R, X_exp)  # (B, T, V, 3, J)
    X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
    X_cam = X_cam + t[..., None, :]
    Z = X_cam[..., 2:3]
    Z_safe = Z.sign() * (Z.abs() + 1e-6)
    X_norm = X_cam / Z_safe
    proj = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)  # (B, T, V, J, 3)
    proj_2d = proj[..., :2] / proj[..., 2:3]
    diff = proj_2d - points_2d  # (B, T, V, J, 2)
    residual = diff.norm(dim=-1)  # (B, T, V, J)
    return residual


def triangulate_dlt_per_joint(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Triangulate each joint independently using weighted DLT.

    Args:
        points_2d: (B, T, V, J, 2).
        proj_matrices: (B, T, V, 3, 4).
        weights: (B, T, V, J).

    Returns:
        X: (B, T, J, 3).
    """
    B, T, V, J, _ = points_2d.shape
    X = torch.zeros(B, T, J, 3, device=points_2d.device, dtype=points_2d.dtype)
    # Reshape to (B*T, V, J, 2) etc.
    pts = points_2d.reshape(B * T, V, J, 2)
    P = proj_matrices.reshape(B * T, V, 3, 4)
    w = weights.reshape(B * T, V, J)
    for j in range(J):
        X[:, :, j, :] = triangulate_dlt_torch(pts[:, :, j, :], P, w[:, :, j]).reshape(B, T, 3)
    return X


class TestTimeSelfEvolutionV27(nn.Module):
    """Iterative geometric self-consistency at inference.

    Parameters
    ----------
    n_iters:
        Maximum number of self-evolution steps.
    residual_thresh_mm:
        Early-stop if mean pose change (mm) falls below this threshold.
    sigma_reproj:
        Scale (pixels) of the Cauchy re-weighting kernel.
    """

    def __init__(
        self,
        n_iters: int = 3,
        residual_thresh_mm: float = 0.5,
        sigma_reproj: float = 5.0,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.residual_thresh_mm = residual_thresh_mm
        self.sigma_reproj = sigma_reproj

    def forward(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine ``pred_3d`` by iterative self-consistency.

        Args:
            pred_3d: (B, T, J, 3) initial 3D pose.
            points_2d: (B, T, V, J, 2) detected 2D keypoints.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            view_mask: optional (B, T, V) bool mask.
            confidence: optional (B, T, V, J) initial confidences.

        Returns:
            pred_3d_ref: (B, T, J, 3) refined 3D pose.
        """
        B, T, V, J, _ = points_2d.shape
        if confidence is None:
            confidence = torch.ones(B, T, V, J, device=pred_3d.device, dtype=pred_3d.dtype)

        pred_current = pred_3d
        P = build_projection_matrix(K, R, t)
        mask = view_mask
        if mask is not None:
            # Ensure confidence is zero for masked-out views.
            confidence = confidence * mask[:, :, :, None].float()

        for _ in range(self.n_iters):
            residual = compute_reprojection_residual(pred_current, points_2d, K, R, t)
            # Cauchy re-weighting: down-weight views with large residuals.
            w = confidence / (1.0 + (residual / self.sigma_reproj) ** 2)
            if mask is not None:
                w = w * mask[:, :, :, None].float()

            pred_next = triangulate_dlt_per_joint(points_2d, P, w)
            # If any joint has all-zero weights, fall back to the current estimate
            # to avoid NaNs from empty least-squares.
            valid = w.sum(dim=2, keepdim=True).clamp(min=1e-6) > 0  # (B, T, 1, J)
            pred_next = torch.where(valid.transpose(-1, -2).expand_as(pred_next), pred_next, pred_current)

            change_mm = (pred_next - pred_current).norm(dim=-1).mean() * 1000.0
            pred_current = pred_next
            if change_mm < self.residual_thresh_mm:
                break

        return pred_current
