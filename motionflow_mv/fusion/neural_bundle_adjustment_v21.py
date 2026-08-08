"""Neural bundle-adjustment layer (v21).

Given an initial 3D skeleton, per-view 2D keypoints, and calibrated cameras,
this layer jointly refines the 3D pose *and* the camera parameters.  It
alternates between an analytic Gauss-Newton structure update and a lightweight
neural camera-correction head driven by per-view reprojection statistics.

The design keeps the analytic geometry differentiable (so the layer can be a
drop-in refinement block) while making the camera updates learnable, avoiding
the instability of full free-form bundle adjustment at initialization.

Safety additions for the v21 diagnosis iteration:
* Structure is refined *before* the camera head sees the points.
* The point update is detached from the camera-head gradient path.
* Camera updates are gated by a residual-improvement test.
* The camera descriptor uses a compact 3-DOF rotation representation.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn

from .differentiable_bundle_adjustment import _project_and_jacobian


def _so3_exp(axis: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map: axis-angle (N, 3) -> rotation (N, 3, 3)."""
    if axis.dim() == 1:
        axis = axis.unsqueeze(0)
    norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    axis_n = axis / norm
    x, y, z = axis_n[..., 0], axis_n[..., 1], axis_n[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    sin_term = torch.sin(norm).unsqueeze(-1)
    cos_term = (1.0 - torch.cos(norm)).unsqueeze(-1)
    R = I + sin_term * K + cos_term * (K @ K)
    # First-order fallback for extremely small angles.
    skew = torch.stack(
        [
            torch.stack([zeros, -axis[..., 2], axis[..., 1]], dim=-1),
            torch.stack([axis[..., 2], zeros, -axis[..., 0]], dim=-1),
            torch.stack([-axis[..., 1], axis[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )
    small_R = I + skew
    return torch.where(norm.squeeze(-1)[..., None, None] < 1e-6, small_R, R)


def _rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors (stable SO(3) log map).

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        aa: (..., 3) axis-angle vectors.  For near-identity matrices the
            magnitude is close to zero.
    """
    # Trace-based angle.
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    # Axis from the skew-symmetric part of R.
    axis = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )

    # Normalise the axis using 2*sin(angle).  For small angles the axis is
    # ill-defined, so the resulting axis-angle magnitude naturally goes to 0.
    sin_angle = torch.sqrt(torch.clamp(1.0 - cos_angle * cos_angle, min=1e-8))
    axis = axis / (2.0 * sin_angle.unsqueeze(-1)).clamp(min=1e-6)

    return axis * angle.unsqueeze(-1)


def _camera_descriptor(
    residual: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Build a per-camera descriptor from reprojection residuals and cameras.

    Args:
        residual: (B, T, V, J, 2) reprojection residual.
        K: (B, T, V, 3, 3) intrinsics.
        R: (B, T, V, 3, 3) rotations.
        t: (B, T, V, 3) translations.
        weights: (B, T, V, J) non-negative weights.

    Returns:
        descriptor: (B, T, V, D) concatenated feature.
    """
    w = weights.unsqueeze(-1) + 1e-8  # (B, T, V, J, 1)
    mean_res = (residual * w).sum(dim=3) / w.sum(dim=3)  # (B, T, V, 2)
    var_res = ((residual - mean_res.unsqueeze(3)) ** 2 * w).sum(dim=3) / w.sum(dim=3)
    std_res = torch.sqrt(var_res + 1e-6)

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    skew = K[..., 0, 1]
    intr = torch.stack([fx, fy, cx, cy, skew], dim=-1)

    # Compact 3-DOF rotation descriptor instead of the full 9-DOF matrix.
    rot_aa = _rotation_matrix_to_axis_angle(R)
    trans = t

    # Also include total per-view weight (proxy for visibility/confidence).
    weight_sum = weights.sum(dim=-1, keepdim=True)  # (B, T, V, 1)

    return torch.cat([mean_res, std_res, intr, rot_aa, trans, weight_sum], dim=-1)


class _CameraCorrectionHead(nn.Module):
    """Neural head that predicts bounded per-camera corrections."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        max_principal_point_update: float,
        max_focal_scale: float,
        max_rotation_rad: float,
        max_translation: float,
    ):
        super().__init__()
        self.max_principal_point_update = max_principal_point_update
        self.max_focal_scale = max_focal_scale
        self.max_rotation_rad = max_rotation_rad
        self.max_translation = max_translation

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 9),
        )
        # Initialise the final layer to zero so the layer starts as identity/no-op.
        # This prevents the neural camera-correction head from perturbing cameras
        # before it has learned to improve them.
        for param in self.mlp[-1].parameters():
            nn.init.zeros_(param)

    def forward(
        self,
        feat: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict and apply bounded camera corrections.

        Returns:
            K_new, R_new, t_new with the same shapes as the inputs.
        """
        out = self.mlp(feat)  # (B, T, V, 9)
        df = out[..., 0]
        dpp = out[..., 1:3]
        daxis = out[..., 3:6]
        dt = out[..., 6:9]

        # Intrinsics.
        scale = 1.0 + torch.tanh(df) * self.max_focal_scale
        K_new = K.clone()
        K_new[..., 0, 0] = K[..., 0, 0] * scale
        K_new[..., 1, 1] = K[..., 1, 1] * scale
        K_new[..., 0, 2] = K[..., 0, 2] + torch.tanh(dpp[..., 0]) * self.max_principal_point_update
        K_new[..., 1, 2] = K[..., 1, 2] + torch.tanh(dpp[..., 1]) * self.max_principal_point_update

        # Rotation via axis-angle exponential map.
        B, T, V, _ = daxis.shape
        axis_update = torch.tanh(daxis) * self.max_rotation_rad  # (B, T, V, 3)
        R_delta = _so3_exp(axis_update.reshape(-1, 3)).reshape(B, T, V, 3, 3)
        R_new = torch.matmul(R_delta, R)

        # Translation.
        t_new = t + torch.tanh(dt) * self.max_translation

        return K_new, R_new, t_new


class NeuralBundleAdjustment(nn.Module):
    """Neural bundle-adjustment layer that refines 3D joints and cameras.

    Parameters
    ----------
    n_iters:
        Number of alternating structure/camera refinement iterations.
    damping:
        Levenberg-Marquardt damping for the point update.
    max_point_update:
        Maximum absolute 3D point update per iteration (meters).
    max_principal_point_update:
        Maximum absolute principal-point correction (pixels).
    max_focal_scale:
        Maximum multiplicative focal-length correction (e.g. 0.05 = 5%%).
    max_rotation_deg:
        Maximum rotation correction per iteration (degrees).
    max_translation:
        Maximum translation correction per iteration (meters).
    camera_hidden:
        Hidden dimension of the per-camera correction MLP.
    warm_start_structure:
        If True, run one structure-only Gauss-Newton step before the first
        camera update.
    gate_camera_update:
        If True, accept a neural camera update only when it does not increase
        the mean reprojection error.  The gate decision is detached so the
        MLP still receives gradients from accepted updates.
    camera_update_tol:
        Tolerance added to the before-error when deciding whether to accept a
        camera update (pixels^2).
    """

    def __init__(
        self,
        n_iters: int = 2,
        damping: float = 1.0,
        max_point_update: float = 0.05,
        max_principal_point_update: float = 10.0,
        max_focal_scale: float = 0.05,
        max_rotation_deg: float = 2.0,
        max_translation: float = 0.1,
        camera_hidden: int = 64,
        warm_start_structure: bool = True,
        gate_camera_update: bool = True,
        camera_update_tol: float = 1.0,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.damping = damping
        self.max_point_update = max_point_update
        self.warm_start_structure = warm_start_structure
        self.gate_camera_update = gate_camera_update
        self.camera_update_tol = camera_update_tol

        in_dim = 16  # see _camera_descriptor
        self.camera_head = _CameraCorrectionHead(
            in_dim=in_dim,
            hidden=camera_hidden,
            max_principal_point_update=max_principal_point_update,
            max_focal_scale=max_focal_scale,
            max_rotation_rad=math.radians(max_rotation_deg),
            max_translation=max_translation,
        )

    def _mean_reproj_error(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar mean reprojection error (pixels^2)."""
        residual, _, valid = _project_and_jacobian(X, points_2d, K, R, t)
        err = (residual ** 2).sum(dim=-1)  # (B, T, V, J)
        mask = weights * valid.float()
        return (err * mask).sum() / mask.sum().clamp(min=1e-6)

    def _update_points(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """One Gauss-Newton/Levenberg-Marquardt step on the 3D structure."""
        residual, J, valid = _project_and_jacobian(X, points_2d, K, R, t)
        w = weights * valid.float()
        w = w / w.sum(dim=2, keepdim=True).clamp(min=1e-6)

        Jw = J * w.unsqueeze(-1).unsqueeze(-1)
        JTJ = torch.einsum("btvjik,btvjil->btjkl", J, Jw)
        JTr = torch.einsum("btvjik,btvji->btjk", J, -(residual * w.unsqueeze(-1)))

        I = torch.eye(3, device=JTJ.device, dtype=JTJ.dtype).view(1, 1, 1, 3, 3)
        delta, *_ = torch.linalg.solve(JTJ + self.damping * I, JTr.unsqueeze(-1))
        delta = delta.squeeze(-1).clamp(-self.max_point_update, self.max_point_update)
        return X + delta

    def _update_cameras(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neural camera-correction step."""
        residual, _, _ = _project_and_jacobian(X, points_2d, K, R, t)
        feat = _camera_descriptor(residual, K, R, t, weights)
        return self.camera_head(feat, K, R, t)

    def forward(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine 3D joints and cameras.

        Args:
            X: (B, T, J, 3) or (B, J, 3) initial 3D joints.
            points_2d: (B, T, V, J, 2) or (B, V, J, 2) observed 2D keypoints.
            K: (B, T, V, 3, 3) or (B, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) or (B, V, 3, 3) rotations.
            t: (B, T, V, 3) or (B, V, 3) translations.
            weights: Optional (B, T, V, J) or (B, V, J) per-view weights.

        Returns:
            X_refined: same shape as input X.
            K_refined: same shape as input K.
            R_refined: same shape as input R.
            t_refined: same shape as input t.
        """
        original_k_dim = K.dim()
        squeeze = False
        if points_2d.dim() == 4:
            points_2d = points_2d.unsqueeze(1)
            X = X.unsqueeze(1)
            squeeze = True

        B, T, V, J, _ = points_2d.shape

        if K.dim() == 3:
            K = K.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            t = t.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
            R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
            t = t.unsqueeze(1).expand(-1, T, -1, -1)

        if weights is None:
            weights = torch.ones(B, T, V, J, device=X.device, dtype=X.dtype)
        elif weights.dim() == 3:
            weights = weights.unsqueeze(1)

        # Ensure X is temporal as well.
        if X.dim() == 3:
            X = X.unsqueeze(1)

        # Optional warm-start: refine structure once before any camera is touched.
        if self.warm_start_structure:
            X = self._update_points(X, points_2d, K, R, t, weights)

        for _ in range(self.n_iters):
            # 1. Refine structure given current cameras.
            X = self._update_points(X, points_2d, K, R, t, weights)

            # 2. Propose neural camera correction.  Detach the point update so
            # the camera head cannot drive gradients through the analytic solver.
            err_before = self._mean_reproj_error(X.detach(), points_2d, K, R, t, weights)
            K_new, R_new, t_new = self._update_cameras(
                X.detach(), points_2d, K, R, t, weights
            )

            if self.gate_camera_update:
                err_after = self._mean_reproj_error(
                    X.detach(), points_2d, K_new, R_new, t_new, weights
                )
                # Accept only if the update does not increase reprojection error.
                use_new = (err_after < err_before + self.camera_update_tol).float()
                # Per-view decision: shape (B, T, V) -> broadcast for K/R/t.
                use_new_k = use_new[..., None, None]
                use_new_t = use_new[..., None]
                K = use_new_k * K_new + (1.0 - use_new_k) * K
                R = use_new_k * R_new + (1.0 - use_new_k) * R
                t = use_new_t * t_new + (1.0 - use_new_t) * t
            else:
                K, R, t = K_new, R_new, t_new

        if squeeze:
            X = X.squeeze(1)
            if original_k_dim in (3, 4):
                K = K[:, 0]
                R = R[:, 0]
                t = t[:, 0]
        elif original_k_dim in (3, 4):
            # Cameras were broadcast over the temporal dimension; collapse it.
            K = K[:, 0]
            R = R[:, 0]
            t = t[:, 0]

        return X, K, R, t


if __name__ == "__main__":
    B, T, V, J = 2, 3, 4, 17
    X = torch.randn(B, T, J, 3) * 0.5 + torch.tensor([0.0, 0.0, 3.0])
    points_2d = torch.randn(B, T, V, J, 2) * 100 + 320
    K = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3).clone()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, V, 3, 3)
    t = torch.zeros(B, T, V, 3)
    weights = torch.ones(B, T, V, J)

    nba = NeuralBundleAdjustment(n_iters=2)
    X_ref, K_ref, R_ref, t_ref = nba(X, points_2d, K, R, t, weights)
    assert X_ref.shape == X.shape
    assert K_ref.shape == K.shape
    assert R_ref.shape == R.shape
    assert t_ref.shape == t.shape
    print("Neural bundle adjustment smoke test passed.")
