"""Multi-view triangulation for 2D keypoints.

Implements Direct Linear Transform (DLT) with optional confidence weighting.
"""

import numpy as np
import torch


def triangulate_dlt_torch(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable/torch DLT triangulation for a single joint.

    Args:
        points_2d: (B, V, 2) or (V, 2) tensor of 2D points.
        proj_matrices: (B, V, 3, 4) or (V, 3, 4) projection matrices.
        weights: optional (B, V) or (V,) non-negative weights.

    Returns:
        X: (B, 3) or (3,) triangulated 3D point(s).
    """
    single = points_2d.dim() == 2
    if single:
        points_2d = points_2d.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)

    B, V, _ = points_2d.shape
    if proj_matrices.dim() == 3:
        proj_matrices = proj_matrices.unsqueeze(0).expand(B, -1, -1, -1)

    if weights is None:
        weights = torch.ones(B, V, device=points_2d.device, dtype=points_2d.dtype)
    else:
        weights = weights.reshape(B, V)

    # Build weighted least-squares system.
    A_list = []
    for v in range(V):
        P = proj_matrices[:, v, :, :]  # (B, 3, 4)
        x = points_2d[:, v, 0]  # (B,)
        y = points_2d[:, v, 1]  # (B,)
        A_list.append(x[:, None, None] * P[:, 2:3, :] - P[:, 0:1, :])
        A_list.append(y[:, None, None] * P[:, 2:3, :] - P[:, 1:2, :])
    A = torch.cat(A_list, dim=1)  # (B, 2V, 4)

    w_sqrt = weights.unsqueeze(-1).repeat(1, 1, 2).view(B, 2 * V, 1)
    A3 = A[..., :3] * torch.sqrt(w_sqrt + 1e-6)
    b = -A[..., 3:] * torch.sqrt(w_sqrt + 1e-6)
    X, *_ = torch.linalg.lstsq(A3, b)
    X = X.squeeze(-1)
    return X.squeeze(0) if single else X


def triangulate_dlt(points_2d: np.ndarray, proj_matrices: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Triangulate one 3D point from N calibrated views.

    Args:
        points_2d: (N, 2) array of 2D keypoints.
        proj_matrices: (N, 3, 4) projection matrices P_i.
        weights: optional (N,) array for confidence weighting.

    Returns:
        (3,) array, the triangulated 3D point in world coordinates.
    """
    points_2d = np.asarray(points_2d, dtype=np.float64)
    proj_matrices = np.asarray(proj_matrices, dtype=np.float64)
    n_views = points_2d.shape[0]

    if weights is None:
        weights = np.ones(n_views, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(n_views)
        weights = np.sqrt(weights + 1e-6)

    A = []
    for (u, v), P, w in zip(points_2d, proj_matrices, weights):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    A = np.stack(A)  # (2*N, 4)

    # Use torch SVD to avoid Windows/OpenBLAS crashes with numpy.linalg.svd.
    A_t = torch.from_numpy(A)
    _, _, Vh = torch.linalg.svd(A_t)
    X = Vh[-1].numpy()
    return (X[:3] / X[3]).astype(np.float64)


def triangulate_confidence_weighted(points_2d: np.ndarray, proj_matrices: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    """Confidence-weighted DLT triangulation.

    Args:
        points_2d: (N, 2)
        proj_matrices: (N, 3, 4)
        confidences: (N,) confidence scores in [0, 1] or any non-negative values.

    Returns:
        (3,) triangulated point.
    """
    return triangulate_dlt(points_2d, proj_matrices, weights=confidences)
