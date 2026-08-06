"""Iteratively reweighted least-squares (IRLS) triangulation with a Charbonnier
loss.

This is a purely geometric baseline: it does not learn any parameters.  Starting
from a confidence-weighted DLT estimate, it reweights views by the inverse
Charbonnier robust loss and re-triangulates for a fixed number of iterations.
Outliers (occluded/noisy views) are automatically down-weighted.

Note:
    The linear-algebra operations are performed with PyTorch (CPU) because the
    Windows numpy/MKL stack in the development environment raises fatal
    exceptions for ``numpy.linalg.lstsq``/``svd``.  The public API remains a
    NumPy interface.
"""

from typing import Optional

import numpy as np
import torch


def _as_tensor(arr: np.ndarray, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(dtype)


def _project_points(X: torch.Tensor, proj_matrices: torch.Tensor) -> torch.Tensor:
    """Project 3D points into 2D image plane.

    Args:
        X: (B, 3) 3D points.
        proj_matrices: (V, 3, 4) projection matrices.

    Returns:
        (B, V, 2) projected 2D points.
    """
    B = X.shape[0]
    V = proj_matrices.shape[0]
    X_h = torch.cat([X, torch.ones((B, 1), dtype=X.dtype, device=X.device)], dim=-1)  # (B, 4)
    P = proj_matrices[None, ...]  # (1, V, 3, 4)
    x_h = (P @ X_h[:, None, :, None]).squeeze(-1)  # (B, V, 3)
    x = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
    return x


def _weighted_dlt_batch(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Confidence/robust weighted DLT for a batch of joints.

    Args:
        points_2d: (B, V, 2)
        proj_matrices: (V, 3, 4)
        weights: (B, V) non-negative weights.

    Returns:
        X: (B, 3)
    """
    B, V, _ = points_2d.shape
    A = torch.zeros((B, 2 * V, 4), dtype=points_2d.dtype, device=points_2d.device)
    for v in range(V):
        P = proj_matrices[v]
        x = points_2d[:, v, 0]
        y = points_2d[:, v, 1]
        A[:, 2 * v] = x[:, None] * P[2:3, :] - P[0:1, :]
        A[:, 2 * v + 1] = y[:, None] * P[2:3, :] - P[1:2, :]

    # Apply weights to both sides.
    w_sqrt = (weights + 1e-9).sqrt().unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * V, 1)
    A = A * w_sqrt
    A3 = A[..., :3]
    b = -A[..., 3:]

    X = torch.linalg.lstsq(A3, b)[0]  # (B, 3, 1)
    return X.squeeze(-1)


def triangulate_irls(
    points_2d: np.ndarray,
    proj_matrices: np.ndarray,
    weights_init: Optional[np.ndarray] = None,
    n_iters: int = 5,
    eps: float = 2.0,
    confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """IRLS triangulation with Charbonnier robust loss.

    Args:
        points_2d: (V, 2) or (B, V, 2) 2D keypoints.
        proj_matrices: (V, 3, 4) projection matrices.
        weights_init: optional (B, V) or (V,) initial robust weights.
        n_iters: number of IRLS iterations.
        eps: Charbonnier epsilon in the same unit as the 2D points (pixels).
        confidences: optional (B, V) or (V,) confidence scores multiplied with
            the robust weights. If None, treated as all ones.

    Returns:
        X: (3,) or (B, 3) triangulated 3D point(s).
    """
    single = points_2d.ndim == 2
    if single:
        points_2d = points_2d[None, ...]  # (1, V, 2)

    points_2d = np.asarray(points_2d, dtype=np.float64)
    proj_matrices = np.asarray(proj_matrices, dtype=np.float64)
    B, V = points_2d.shape[:2]

    if confidences is None:
        confidences = np.ones((B, V), dtype=np.float64)
    else:
        confidences = np.asarray(confidences, dtype=np.float64).reshape(B, V)

    if weights_init is None:
        weights = confidences.copy()
    else:
        weights = np.asarray(weights_init, dtype=np.float64).reshape(B, V) * confidences

    p2d_t = _as_tensor(points_2d)
    P_t = _as_tensor(proj_matrices)
    conf_t = _as_tensor(confidences)
    weights_t = _as_tensor(weights)

    for _ in range(n_iters):
        X = _weighted_dlt_batch(p2d_t, P_t, weights_t)
        x_pred = _project_points(X, P_t)  # (B, V, 2)
        residuals = (x_pred - p2d_t).norm(dim=-1)  # (B, V)
        weights_t = conf_t / (residuals ** 2 + eps ** 2).sqrt()

    X_np = X.detach().cpu().numpy()
    if single:
        return X_np[0]
    return X_np
