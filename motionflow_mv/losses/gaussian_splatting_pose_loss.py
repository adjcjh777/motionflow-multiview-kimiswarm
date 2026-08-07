"""Gaussian-splatting pose regularizer."""

import torch


def _robust_cholesky(A: torch.Tensor, max_jitter: float = 1.0) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError:
        pass
    jitter = 1e-4
    while jitter <= max_jitter:
        A_jittered = A + jitter * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).view(
            1, 1, 1, 1, A.shape[-1], A.shape[-1]
        )
        try:
            return torch.linalg.cholesky(A_jittered)
        except RuntimeError:
            jitter *= 10.0
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
    log_std_view: torch.Tensor = None,
) -> torch.Tensor:
    """Negative log-likelihood of 2-D observations under projected 3-D Gaussians."""
    if pred_3d.dim() == 4:
        B, T, J, _ = pred_3d.shape
    else:
        raise ValueError("pred_3d must have shape (B, T, J, 3)")

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

    if log_std_view is not None:
        log_std_total = log_std.unsqueeze(2) + log_std_view
        std = log_std_total.exp().clamp(min=0.01, max=10.0)
        Sigma_3d = torch.diag_embed(std ** 2)
    else:
        std = log_std.exp().clamp(min=0.01, max=10.0)
        Sigma_3d = torch.diag_embed(std ** 2)

    if log_std_view is not None:
        Sigma_2d = J @ Sigma_3d @ J.transpose(-2, -1)
    else:
        Sigma_2d = J @ Sigma_3d.unsqueeze(2) @ J.transpose(-2, -1)
    Sigma_2d = Sigma_2d + eps * torch.eye(2, device=Sigma_2d.device, dtype=Sigma_2d.dtype).view(
        1, 1, 1, 1, 2, 2
    )

    diff = (points_2d - proj).unsqueeze(-1)
    L = _robust_cholesky(Sigma_2d)
    y_solve = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis_sq = (y_solve ** 2).sum(dim=(-2, -1))

    logdet = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1).abs()).sum(dim=-1)

    nll = 0.5 * (logdet + mahalanobis_sq)

    if confidences is not None:
        weight = confidences
    else:
        weight = torch.ones_like(nll)

    loss = (nll * weight).sum() / (weight.sum() + eps)

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
    L = _robust_cholesky(Sigma_2d)
    y_solve = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis = (y_solve ** 2).sum(dim=(-2, -1)).sqrt()
    return mahalanobis
