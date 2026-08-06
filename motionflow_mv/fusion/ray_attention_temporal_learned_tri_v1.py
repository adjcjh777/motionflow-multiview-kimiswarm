"""Temporal ray-attention fusion with a learnable Gauss-Newton triangulation head.

Extends ``RayAttentionFusionModelTemporalResidual`` by replacing the weighted
DLT triangulation step with a differentiable Gauss-Newton optimizer that
refines the 3D estimate using the network-predicted per-view weights.  The
residual refinement head is kept on top of the GN-refined estimate.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    X: (B, T, J, 3) refined world-coordinate 3D joints, or (B, J, 3) for 4D input
    weights: (B, T, V, J) predicted per-view per-joint weights, or (B, V, J)
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .ray_attention_model import _triangulate_weighted_dlt


def _triangulate_weighted_gauss_newton(
    points_2d: torch.Tensor,
    weights: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    init_3d: torch.Tensor,
    num_iters: int = 3,
    damping: float = 1e-6,
) -> torch.Tensor:
    """Differentiable weighted Gauss-Newton triangulation.

    Refines an initial 3D estimate by minimizing the weighted reprojection
    error:

        E(X) = sum_v w_v * || pi_v(X) - x_v ||^2,

    where pi_v is the projection for view v.  The solver is fully differentiable,
    so gradients flow back through the predicted weights and the camera parameters.

    Args:
        points_2d: (N, V, J, 2)
        weights: (N, V, J)
        K: (N, V, 3, 3)
        R: (N, V, 3, 3)
        t: (N, V, 3)
        init_3d: (N, J, 3) initial estimate (typically from DLT)
        num_iters: number of Gauss-Newton iterations
        damping: diagonal damping added to the normal equations for stability

    Returns:
        X: (N, J, 3)
    """
    N, V, J, _ = points_2d.shape
    X = init_3d  # (N, J, 3)

    # Extract intrinsics.  K is assumed to be an upper-triangular pinhole matrix.
    fx = K[:, :, 0, 0]  # (N, V)
    s = K[:, :, 0, 1]
    cx = K[:, :, 0, 2]
    fy = K[:, :, 1, 1]
    cy = K[:, :, 1, 2]

    eye3 = torch.eye(3, device=X.device, dtype=X.dtype).view(1, 1, 3, 3)

    for _ in range(max(1, num_iters)):
        # Camera coordinates: (N, V, J, 3)
        X_cam = torch.einsum("nvab,njb->nvja", R, X) + t.unsqueeze(2)
        x_c = X_cam[..., 0]
        y_c = X_cam[..., 1]
        z_c = X_cam[..., 2]

        inv_z = 1.0 / (z_c + 1e-8)
        u = (fx[:, :, None] * x_c + s[:, :, None] * y_c + cx[:, :, None] * z_c) * inv_z
        v = (fy[:, :, None] * y_c + cy[:, :, None] * z_c) * inv_z
        proj = torch.stack([u, v], dim=-1)  # (N, V, J, 2)
        r = points_2d - proj  # (N, V, J, 2)

        # Jacobian of projection w.r.t. camera coordinates (N, V, J, 2, 3).
        J_cam = torch.zeros(N, V, J, 2, 3, device=X.device, dtype=X.dtype)
        J_cam[:, :, :, 0, 0] = fx[:, :, None] * inv_z
        J_cam[:, :, :, 0, 1] = s[:, :, None] * inv_z
        J_cam[:, :, :, 0, 2] = (cx[:, :, None] - u) * inv_z
        J_cam[:, :, :, 1, 1] = fy[:, :, None] * inv_z
        J_cam[:, :, :, 1, 2] = (cy[:, :, None] - v) * inv_z

        # Chain rule: w.r.t. world coordinates.
        J_world = torch.einsum("nvjab,nvbd->nvjad", J_cam, R)  # (N, V, J, 2, 3)

        # Flatten views/residuals per joint.
        J_world = J_world.permute(0, 2, 1, 3, 4).reshape(N, J, V * 2, 3)  # (N, J, 2V, 3)
        r_flat = r.permute(0, 2, 1, 3).reshape(N, J, V * 2)  # (N, J, 2V)
        w_flat = (
            weights.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, 2)
            .reshape(N, J, V * 2)
        )  # (N, J, 2V)

        # Weighted normal equations.
        A = torch.einsum(
            "njkp,njkq->njpq",
            J_world,
            J_world * w_flat[..., None],
        )  # (N, J, 3, 3)
        b = torch.einsum(
            "njkp,njk->njp",
            J_world,
            r_flat * w_flat,
        )  # (N, J, 3)

        A = A + damping * eye3.expand(N, J, -1, -1)

        # Solve for update and apply.
        b = b.unsqueeze(-1)  # (N, J, 3, 1)
        dx = torch.linalg.solve(A, b).squeeze(-1)  # (N, J, 3)
        X = X + dx

    return X


class RayAttentionFusionModelTemporalResidualLearnedTri(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-attention fusion with a learnable Gauss-Newton triangulation head.

    The model predicts per-view weights, uses DLT for an initial 3D estimate,
    then refines that estimate with a differentiable Gauss-Newton solver
    operating on the weighted reprojection error.  The existing residual MLP
    still operates on top of the refined triangulated output.

    Parameters
    ----------
    gn_iters:
        Number of Gauss-Newton iterations (default 3).
    gn_damping:
        Diagonal damping added to the GN normal equations (default 1e-6).
    **kwargs:
        Passed to ``RayAttentionFusionModelTemporalResidual``.
    """

    def __init__(self, gn_iters: int = 3, gn_damping: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.gn_iters = gn_iters
        self.gn_damping = gn_damping

    def _triangulate(
        self,
        points_2d: torch.Tensor,
        weights: torch.Tensor,
        P: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        init_3d = _triangulate_weighted_dlt(points_2d, weights, P)
        return _triangulate_weighted_gauss_newton(
            points_2d,
            weights,
            K,
            R,
            t,
            init_3d,
            num_iters=self.gn_iters,
            damping=self.gn_damping,
        )


if __name__ == "__main__":
    # Shape/gradient sanity check.
    import numpy as np
    from ..calibration.camera import Camera

    def _make_cameras(n_views: int = 4):
        cameras = []
        for i in range(n_views):
            theta = 2 * np.pi * i / n_views
            c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
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
            cameras.append(Camera(K=K, R=R, t=t))
        return cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalResidualLearnedTri(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("temporal learned-triangulation model sanity check passed")
