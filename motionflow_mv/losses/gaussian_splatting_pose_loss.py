"""Gaussian-splatting pose regularizer.

Projects each predicted 3-D joint, represented as an anisotropic 3-D Gaussian,
into every calibrated view and evaluates the negative log-likelihood of the
observed 2-D keypoints under the projected 2-D Gaussian.  This acts as a
cross-view geometric consistency term and a learned per-joint uncertainty model.
"""

import torch
import torch.nn.functional as F


def _robust_cholesky(A: torch.Tensor, max_jitter: float = 1.0) -> torch.Tensor:
    """Cholesky with jitter fallback for numerical stability."""
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError:
        pass
    # Add increasing diagonal jitter until positive-definite.
    jitter = 1e-4
    while jitter <= max_jitter:
        A_jittered = A + jitter * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).view(
            1, 1, 1, 1, A.shape[-1], A.shape[-1]
        )
        try:
            return torch.linalg.cholesky(A_jittered)
        except RuntimeError:
            jitter *= 10.0
    # Fallback: eigendecomposition with clipped eigenvalues.
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp(min=1e-6)
    A_pd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    return torch.linalg.cholesky(A_pd)


def gaussian_splatting_pose_loss(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    log_std: torch.Tensor,
    confidences: torch.Tensor = None,
    eps: float = 1e-4,
    trace_weight: float = 1e-4,
) -> torch.Tensor:
    """Negative log-likelihood of 2-D observations under projected 3-D Gaussians.

    Parameters
    ----------
    pred_3d: (B, T, J, 3)
        Predicted world 3-D joints.
    points_2d: (B, T, V, J, 2)
        Observed 2-D keypoints.
    K: (B, V, 3, 3)
        Intrinsic matrices.
    R: (B, V, 3, 3)
        Rotation matrices (world -> camera).
    t: (B, V, 3)
        Translation vectors (world -> camera).
    log_std: (B, T, J, 3)
        Log standard deviations for the per-joint anisotropic 3-D Gaussian.
    confidences: (B, T, V, J), optional
        Per-observation weights in [0, 1].
    eps:
        Small constant for numerical stability.
    trace_weight:
        Weight for the covariance-trace regularizer.

    Returns
    -------
    loss:
        Scalar tensor.
    """
    if pred_3d.dim() == 4:
        B, T, J, _ = pred_3d.shape
    else:
        raise ValueError("pred_3d must have shape (B, T, J, 3)")

    V = K.shape[1]

    # Camera coordinates.
    X = pred_3d.unsqueeze(2)  # (B, T, 1, J, 3)
    R_exp = R.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3, 3)
    t_exp = t.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3)
    X_cam = (R_exp @ X.unsqueeze(-1)).squeeze(-1) + t_exp  # (B, T, V, J, 3)

    x = X_cam[..., 0]
    y = X_cam[..., 1]
    z = X_cam[..., 2]

    # Projected 2-D mean.
    fx = K[:, None, :, None, 0, 0]
    fy = K[:, None, :, None, 1, 1]
    cx = K[:, None, :, None, 0, 2]
    cy = K[:, None, :, None, 1, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy
    proj = torch.stack([u, v], dim=-1)  # (B, T, V, J, 2)

    # Jacobian of projection w.r.t. camera coordinates.
    z_sq = z ** 2
    zero = torch.zeros_like(x)
    J_uv = torch.stack(
        [
            torch.stack([fx / z, zero, -fx * x / z_sq], dim=-1),
            torch.stack([zero, fy / z, -fy * y / z_sq], dim=-1),
        ],
        dim=-2,
    )  # (B, T, V, J, 2, 3)

    # Chain rule with rotation.
    J = J_uv @ R_exp  # (B, T, V, J, 2, 3)

    # 3-D covariance from predicted log std.
    std = log_std.exp().clamp(min=0.01, max=10.0)  # (B, T, J, 3)
    Sigma_3d = torch.diag_embed(std ** 2)  # (B, T, J, 3, 3)

    # Project to 2-D.
    Sigma_2d = J @ Sigma_3d.unsqueeze(2) @ J.transpose(-2, -1)
    Sigma_2d = Sigma_2d + eps * torch.eye(2, device=Sigma_2d.device, dtype=Sigma_2d.dtype).view(
        1, 1, 1, 1, 2, 2
    )

    # Difference between observation and projected mean.
    diff = (points_2d - proj).unsqueeze(-1)  # (B, T, V, J, 2, 1)

    # Cholesky solve for Mahalanobis distance and log determinant.
    # Try with increasingly large jitter if the matrix is not positive-definite.
    L = _robust_cholesky(Sigma_2d)
    y_solve = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis_sq = (y_solve ** 2).sum(dim=(-2, -1))  # (B, T, V, J)

    logdet = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1).abs()).sum(dim=-1)  # (B, T, V, J)

    nll = 0.5 * (logdet + mahalanobis_sq)

    if confidences is not None:
        weight = confidences
    else:
        weight = torch.ones_like(nll)

    loss = (nll * weight).sum() / (weight.sum() + eps)

    # Regularize covariance magnitude.
    trace = std.sum(dim=-1).mean()
    loss = loss + trace_weight * trace

    return loss


def gaussian_splatting_render_error(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Compute per-joint/view reprojection Mahalanobis distance (diagnostic only).

    Returns
    -------
    error: (B, T, V, J)
        Per-joint per-view Mahalanobis distance.
    """
    B, T, J, _ = pred_3d.shape
    V = K.shape[1]

    X = pred_3d.unsqueeze(2)
    R_exp = R.unsqueeze(1).unsqueeze(3)
    t_exp = t.unsqueeze(1).unsqueeze(3)
    X_cam = (R_exp @ X.unsqueeze(-1)).squeeze(-1) + t_exp

    x = X_cam[..., 0]
    y = X_cam[..., 1]
    z = X_cam[..., 2]

    fx = K[:, None, :, None, 0, 0]
    fy = K[:, None, :, None, 1, 1]
    cx = K[:, None, :, None, 0, 2]
    cy = K[:, None, :, None, 1, 2]

    u = fx * x / z + cx
    v = fy * y / z + cy
    proj = torch.stack([u, v], dim=-1)

    z_sq = z ** 2
    zero = torch.zeros_like(x)
    J_uv = torch.stack(
        [
            torch.stack([fx / z, zero, -fx * x / z_sq], dim=-1),
            torch.stack([zero, fy / z, -fy * y / z_sq], dim=-1),
        ],
        dim=-2,
    )
    J = J_uv @ R_exp

    std = log_std.exp().clamp(min=0.01, max=10.0)
    Sigma_3d = torch.diag_embed(std ** 2)
    Sigma_2d = J @ Sigma_3d.unsqueeze(2) @ J.transpose(-2, -1)
    Sigma_2d = Sigma_2d + 1e-4 * torch.eye(2, device=Sigma_2d.device, dtype=Sigma_2d.dtype).view(
        1, 1, 1, 1, 2, 2
    )

    diff = (points_2d - proj).unsqueeze(-1)
    L = torch.linalg.cholesky(Sigma_2d)
    y_solve = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis = (y_solve ** 2).sum(dim=(-2, -1)).sqrt()
    return mahalanobis
