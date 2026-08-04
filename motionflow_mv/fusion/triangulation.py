"""Multi-view triangulation for 2D keypoints.

Implements Direct Linear Transform (DLT) with optional confidence weighting.
"""

import numpy as np


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

    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
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
