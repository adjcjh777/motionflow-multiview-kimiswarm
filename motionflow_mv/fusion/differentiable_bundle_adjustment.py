"""Differentiable structure-only bundle adjustment for calibrated multi-view pose.

Given an initial 3D skeleton and a set of calibrated views, this module
performs a small number of Gauss-Newton/Levenberg-Marquardt steps that refine
the 3D structure by minimizing the weighted reprojection residual.

Cameras (intrinsics and extrinsics) are kept fixed; only the 3D joint
positions are updated.  This makes the module a drop-in refinement layer for
any calibrated multi-view fusion model.
"""

import torch
import torch.nn as nn


def _ensure_5d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Add a temporal dimension if needed; return (B, T, J, 3) and a flag."""
    if x.dim() == 3:  # (B, J, 3)
        return x.unsqueeze(1), True
    return x, False


def _project_and_jacobian(
    X: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reprojection residuals and analytic 2x3 Jacobians.

    Args:
        X: (B, T, J, 3) world points.
        points_2d: (B, T, V, J, 2) observed image points.
        K: (B, T, V, 3, 3) intrinsics.
        R: (B, T, V, 3, 3) rotations (world -> camera).
        t: (B, T, V, 3) translations (world -> camera).

    Returns:
        residuals: (B, T, V, J, 2) projected - observed.
        J: (B, T, V, J, 2, 3) Jacobian of residual w.r.t. X.
        valid: (B, T, V, J) boolean mask of points with positive depth.
    """
    # Expand X over views: (B, T, J, 3) -> (B, T, V, J, 3).
    X_expanded = X.unsqueeze(2)

    # Transform to camera coordinates.
    X_cam = torch.matmul(R.unsqueeze(3), X_expanded.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    X_cam = X_cam + t.unsqueeze(3)  # broadcast translation over J
    z = X_cam[..., 2]  # (B, T, V, J)
    z_safe = z.clamp(min=1e-6)

    # Project.
    proj = torch.matmul(K.unsqueeze(3), X_cam.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
    uv = proj[..., :2] / z_safe.unsqueeze(-1)  # (B, T, V, J, 2)

    residual = uv - points_2d  # (B, T, V, J, 2)

    # Analytic Jacobian of [u, v] w.r.t. X.
    # For each view/joint, J_uv = d(uv)/d(X_cam) @ R.
    # d(uv)/d(X_cam) in camera coordinates is:
    #   [fx * (1/z,      0, -x/z^2),
    #    [0,       fy/z, -y/z^2]]
    fx = K[..., 0, 0].unsqueeze(-1)  # (B, T, V, 1)
    fy = K[..., 1, 1].unsqueeze(-1)
    x = X_cam[..., 0]
    y = X_cam[..., 1]
    z2 = z_safe * z_safe

    # d(uv)/d(X_cam) has shape (B, T, V, J, 2, 3)
    J_cam = torch.zeros(*X_cam.shape[:-1], 2, 3, device=X.device, dtype=X.dtype)
    J_cam[..., 0, 0] = fx / z_safe
    J_cam[..., 0, 2] = -fx * x / z2
    J_cam[..., 1, 1] = fy / z_safe
    J_cam[..., 1, 2] = -fy * y / z2

    # J = J_cam @ R -> (B, T, V, J, 2, 3)
    J = torch.matmul(J_cam, R.unsqueeze(3))

    valid = z > 1e-4
    return residual, J, valid


class DifferentiableBundleAdjustment(nn.Module):
    """Structure-only differentiable bundle adjustment.

    Parameters
    ----------
    n_iters:
        Number of Gauss-Newton/Levenberg-Marquardt iterations (default 2).
    damping:
        Levenberg-Marquardt damping factor (default 1.0).
    max_update:
        Maximum absolute 3D update in meters (default 0.05).
    """

    def __init__(
        self,
        n_iters: int = 2,
        damping: float = 1.0,
        max_update: float = 0.05,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.damping = damping
        self.max_update = max_update

    def forward(
        self,
        X: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Refine 3D points by minimizing weighted reprojection error.

        Args:
            X: (B, T, J, 3) or (B, J, 3) initial 3D points.
            points_2d: (B, T, V, J, 2) or (B, V, J, 2) observed 2D keypoints.
            K: (B, T, V, 3, 3) or (B, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) or (B, V, 3, 3) rotations.
            t: (B, T, V, 3) or (B, V, 3) translations.
            weights: (B, T, V, J) or (B, V, J) non-negative per-view weights.

        Returns:
            X_refined: same shape as input X.
        """
        squeeze = False
        X, squeeze = _ensure_5d(X, "X")
        if points_2d.dim() == 4:
            points_2d = points_2d.unsqueeze(1)
        if K.dim() == 4:
            K = K.unsqueeze(1)
            R = R.unsqueeze(1)
            t = t.unsqueeze(1)
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)

        B, T, V, J = points_2d.shape[:4]
        X_ref = X.clone()

        for _ in range(self.n_iters):
            residual, J, valid = _project_and_jacobian(X_ref, points_2d, K, R, t)
            # residual, J: (B, T, V, J, 2), (B, T, V, J, 2, 3)
            # weights: (B, T, V, J)
            w = weights * valid.float()
            w = w / (w.sum(dim=2, keepdim=True).clamp(min=1e-6))  # normalize over views
            # Solve weighted least-squares per joint.
            # For each joint, we have 2V residuals and a 2V x 3 Jacobian.
            # We solve (J^T W J + lambda I) delta = -J^T W r.
            Jw = J * w.unsqueeze(-1).unsqueeze(-1)  # (B, T, V, J, 2, 3)
            JTJ = torch.einsum("btvjik,btvjil->btjkl", J, Jw)  # (B, T, J, 3, 3)
            JTr = torch.einsum("btvjik,btvji->btjk", J, -residual * w.unsqueeze(-1))  # (B, T, J, 3)

            # Levenberg-Marquardt damping.
            I = torch.eye(3, device=JTJ.device, dtype=JTJ.dtype).view(1, 1, 1, 3, 3)
            JTJ_damped = JTJ + self.damping * I

            # Solve for update.
            # delta = (JTJ + lambda I)^{-1} JTr
            delta, *_ = torch.linalg.solve(JTJ_damped, JTr.unsqueeze(-1))
            delta = delta.squeeze(-1).clamp(-self.max_update, self.max_update)

            X_ref = X_ref + delta

        if squeeze:
            X_ref = X_ref.squeeze(1)
        return X_ref


if __name__ == "__main__":
    torch.manual_seed(0)
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

    dba = DifferentiableBundleAdjustment(n_iters=2)
    X_ref = dba(X, points_2d, K, R, t, weights)
    assert X_ref.shape == X.shape
    print("DBA module smoke test passed.")
