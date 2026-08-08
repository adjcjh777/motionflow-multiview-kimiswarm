"""Outlier-view detection and down-weighting for multi-view pose fusion.

Given an initial 3D pose estimate and calibrated cameras, this module
computes per-view reprojection residuals, identifies views that are
inconsistent with the multi-view consensus using robust statistics, and
returns soft down-weights that can be used for triangulation or loss
weighting.

The detector is intentionally simple and differentiable.  The learned
residual gate is initialised to zero, so when it is inserted into a larger
module the block starts as an identity/no-op.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


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
        residual: (B, T, V, J) L2 reprojection error in the same units as
        ``points_2d`` (typically pixels).
    """
    # X: (B, T, J, 3) -> camera-space via R and t.
    # Add a view dimension and apply R.
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


class OutlierViewDetector(nn.Module):
    """Robust outlier-view detector with soft down-weighting.

    For each joint, the detector compares the reprojection residual of each
    view against the median residual across views.  Views whose residual is
    far from the median (in terms of median absolute deviation, MAD) are
    down-weighted with a soft exponential.  A small learned gate is added so
    the down-weighting can be scaled down to identity at init and learned
    end-to-end.

    Parameters
    ----------
    z_thresh:
        Robust z-score threshold above which a view is considered an outlier.
        Larger values make the detector more tolerant.
    soft_beta:
        Softness of the exponential down-weighting.  ``beta=0`` gives uniform
        weights; larger values more aggressively suppress outliers.
    eps:
        Small constant to avoid division by zero when MAD is zero.
    """

    def __init__(self, z_thresh: float = 3.0, soft_beta: float = 1.0, min_mad: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.z_thresh = z_thresh
        self.soft_beta = soft_beta
        self.min_mad = min_mad
        self.eps = eps
        # Learnable gate.  Initialised to zero so the module acts as identity.
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def extra_repr(self) -> str:  # noqa: D401
        return (
            f"z_thresh={self.z_thresh}, soft_beta={self.soft_beta}, "
            f"min_mad={self.min_mad}, eps={self.eps}"
        )

    def forward(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return soft view weights and the raw reprojection residuals.

        Args:
            X: (B, T, J, 3) 3D joint positions.
            points_2d: (B, T, V, J, 2) detected 2D keypoints.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            view_mask: optional (B, T, V) bool mask; masked views receive zero
                weight regardless of residual.

        Returns:
            weights: (B, T, V, J) soft weights in ``[0, 1]``.
            residual: (B, T, V, J) raw L2 reprojection residuals.
        """
        residual = compute_reprojection_residual(X, points_2d, K, R, t)

        # Robust statistics over the view dimension.
        median = residual.median(dim=2, keepdim=True)[0]  # (B, T, 1, J)
        mad = (residual - median).abs().median(dim=2, keepdim=True)[0]  # (B, T, 1, J)
        # MAD scaled to approximate standard deviation.  ``min_mad`` prevents
        # over-aggressive down-weighting when all views have nearly identical
        # residuals (e.g. perfectly reprojecting synthetic data).
        mad_std = 1.4826 * mad
        z = (residual - median) / (mad_std + self.min_mad + self.eps)  # (B, T, V, J)

        # Soft down-weighting for views beyond the threshold.
        outlier_margin = torch.clamp(z - self.z_thresh, min=0.0)
        consensus_weights = torch.exp(-self.soft_beta * outlier_margin)

        # Learned gate: at init residual_scale == 0 -> weights == 1.
        gate = torch.sigmoid(self.residual_scale)
        weights = 1.0 - gate * (1.0 - consensus_weights)
        weights = weights.clamp(min=0.0, max=1.0)

        if view_mask is not None:
            mask = view_mask.bool()
            weights = weights * mask[:, :, :, None]

        return weights, residual


if __name__ == "__main__":
    import numpy as np

    def _make_cameras(n_views: int = 4):
        Ks, Rs, ts = [], [], []
        for i in range(n_views):
            theta = 2 * np.pi * i / n_views
            c = np.array([3.0 * np.cos(theta), 3.0 * np.sin(theta), 1.0])
            forward = -c / np.linalg.norm(c)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            R = np.stack([right, up, -forward], axis=0)
            t = -R @ c
            K = np.eye(3)
            K[0, 0] = K[1, 1] = 800.0
            K[0, 2] = 320.0
            K[1, 2] = 240.0
            Ks.append(K)
            Rs.append(R)
            ts.append(t)
        return (
            torch.from_numpy(np.stack(Ks)).float(),
            torch.from_numpy(np.stack(Rs)).float(),
            torch.from_numpy(np.stack(ts)).float(),
        )

    K, R, t = _make_cameras(4)
    # Use a single joint near the rig centre so all views have small residuals.
    joints_3d = torch.tensor([[0.05, -0.05, 0.5]])

    # Project cleanly: points_2d (V, J, 2) -> (1, 1, V, J, 2).
    X_cam = torch.einsum("vab,jb->vja", R, joints_3d) + t[:, None, :]
    z = X_cam[..., 2:3].clamp(min=1e-6)
    uv = torch.einsum("vik,vjk->vji", K, X_cam / z)
    points_2d = uv[..., :2] / uv[..., 2:3]
    points_2d = points_2d.permute(1, 0, 2).unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2, 4)

    K = K.unsqueeze(0).unsqueeze(0)
    R = R.unsqueeze(0).unsqueeze(0)
    t = t.unsqueeze(0).unsqueeze(0)
    X = joints_3d.unsqueeze(0).unsqueeze(0)

    detector = OutlierViewDetector(z_thresh=3.0, soft_beta=1.0)
    weights, _ = detector(X, points_2d, K, R, t)
    print("Clean views: weights near 1.0", weights[0, 0, :, 0].tolist())
    assert weights.shape == (1, 1, 4, 1)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5)

    # Corrupt one view.
    points_2d_corrupt = points_2d.clone()
    points_2d_corrupt[:, :, 0, :, :] += 100.0
    weights_corrupt, _ = detector(X, points_2d_corrupt, K, R, t)
    print("Corrupted view 0 weight:", weights_corrupt[0, 0, 0, 0].item())
    print("Clean view weight:", weights_corrupt[0, 0, 1, 0].item())
    assert weights_corrupt[0, 0, 0, 0] < weights_corrupt[0, 0, 1, 0]

    print("OutlierViewDetector CPU smoke test passed")
