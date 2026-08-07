"""Differentiable uncertainty-weighted multi-view triangulation.

Given per-view 2D keypoints and calibrated cameras, this module solves the
triangulation problem while weighting each view by its estimated uncertainty
(covariance) in image space.  The weighting is equivalent to using the
precision (inverse covariance) as a Mahalanobis metric in the DLT normal
equations, and the whole pipeline is differentiable w.r.t. the input points,
projection matrices, and uncertainties.

References
----------
Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2004.
"""

from typing import Optional

import torch
import torch.nn as nn


def _build_dlt_matrix(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
) -> torch.Tensor:
    """Build the DLT constraint matrix A of shape (B, 2V, 4).

    Args:
        points_2d: (B, V, 2)
        proj_matrices: (B, V, 3, 4)

    Returns:
        A: (B, 2V, 4)
    """
    B, V, _ = points_2d.shape
    A_list = []
    for v in range(V):
        P = proj_matrices[:, v, :, :]  # (B, 3, 4)
        x = points_2d[:, v, 0]  # (B,)
        y = points_2d[:, v, 1]  # (B,)
        A_list.append(x[:, None, None] * P[:, 2:3, :] - P[:, 0:1, :])
        A_list.append(y[:, None, None] * P[:, 2:3, :] - P[:, 1:2, :])
    A = torch.cat(A_list, dim=1)  # (B, 2V, 4)
    return A


def _weight_matrix_from_uncertainty(
    covariances: Optional[torch.Tensor] = None,
    precisions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weight matrices W such that W^T W equals the precision matrix.

    Supports either covariance or precision input.  For a 2x2 covariance Σ,
    we Cholesky factor it as Σ = S S^T, invert S, and return W = S^{-1},
    giving W^T W = S^{-T} S^{-1} = Σ^{-1}.  For a precision matrix Λ = L L^T,
    we return W = L^T so that W^T W = L L^T = Λ.

    Args:
        covariances: (..., 2, 2) symmetric positive-definite matrices.
        precisions: (..., 2, 2) symmetric positive-definite matrices.

    Returns:
        W: (..., 2, 2) weight matrices.
    """
    if precisions is not None:
        # Precision Λ = L L^T (L lower triangular). W = L^T.
        L = torch.linalg.cholesky(precisions)
        W = L.transpose(-2, -1)
    elif covariances is not None:
        # Covariance Σ = S S^T. W = S^{-1} gives W^T W = Σ^{-1}.
        S = torch.linalg.cholesky(covariances)
        shape = covariances.shape
        eye = torch.eye(
            2,
            device=covariances.device,
            dtype=covariances.dtype,
        )
        eye = eye.view(*([1] * (len(shape) - 2)), 2, 2).expand(*shape[:-2], 2, 2)
        W = torch.linalg.solve_triangular(S, eye, upper=False)
    else:
        raise ValueError("Either covariances or precisions must be provided.")
    return W


def _solve_dlt(
    A: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve the inhomogeneous DLT system with optional per-row weights.

    Args:
        A: (B, M, 4) DLT matrix.
        weights: (B, M, 1) optional positive weights.

    Returns:
        X: (B, 3) triangulated points.
    """
    if weights is not None:
        A = A * weights
    A3 = A[..., :3]  # (B, M, 3)
    b = -A[..., 3:]  # (B, M, 1)
    X, *_ = torch.linalg.lstsq(A3, b)
    return X.squeeze(-1)  # (B, 3)


def triangulate_uncertainty_weighted(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
    precisions: Optional[torch.Tensor] = None,
    confidences: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable uncertainty-weighted DLT triangulation for a single joint.

    Args:
        points_2d: (B, V, 2) or (V, 2) tensor of 2D points.
        proj_matrices: (B, V, 3, 4) or (V, 3, 4) projection matrices.
        covariances: optional (B, V, 2, 2) per-view 2D covariance matrices.
        precisions: optional (B, V, 2, 2) per-view 2D precision matrices.
        confidences: optional (B, V) scalar weights in [0, 1].

    Returns:
        X: (B, 3) or (3,) triangulated 3D point(s).
    """
    single = points_2d.dim() == 2
    if single:
        points_2d = points_2d.unsqueeze(0)
        if covariances is not None:
            covariances = covariances.unsqueeze(0)
        if precisions is not None:
            precisions = precisions.unsqueeze(0)
        if confidences is not None:
            confidences = confidences.unsqueeze(0)

    B, V, _ = points_2d.shape
    if proj_matrices.dim() == 3:
        proj_matrices = proj_matrices.unsqueeze(0).expand(B, -1, -1, -1)

    A = _build_dlt_matrix(points_2d, proj_matrices)  # (B, 2V, 4)

    if covariances is not None or precisions is not None:
        W = _weight_matrix_from_uncertainty(covariances, precisions)  # (B, V, 2, 2)
        A_view = A.reshape(B, V, 2, 4)
        A_weighted = torch.einsum("bvij,bvjk->bvik", W, A_view)  # (B, V, 2, 4)
        A = A_weighted.reshape(B, 2 * V, 4)

    if confidences is not None:
        c = confidences.reshape(B, V, 1).sqrt()
        c = c.repeat(1, 1, 2).reshape(B, 2 * V, 1)
        A = A * c

    X = _solve_dlt(A)
    return X.squeeze(0) if single else X


def triangulate_uncertainty_weighted_batched(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
    precisions: Optional[torch.Tensor] = None,
    confidences: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable uncertainty-weighted DLT triangulation for multiple joints.

    Args:
        points_2d: (N, V, J, 2) tensor of 2D keypoints.
        proj_matrices: (N, V, 3, 4) or (V, 3, 4) projection matrices.
        covariances: optional (N, V, J, 2, 2) per-view 2D covariance matrices.
        precisions: optional (N, V, J, 2, 2) per-view 2D precision matrices.
        confidences: optional (N, V, J) scalar weights in [0, 1].

    Returns:
        X: (N, J, 3) triangulated 3D points.
    """
    if points_2d.dim() != 4:
        raise ValueError(
            f"points_2d must be 4-D (N, V, J, 2), got shape {tuple(points_2d.shape)}"
        )

    N, V, J, _ = points_2d.shape
    points_flat = points_2d.permute(0, 2, 1, 3).reshape(N * J, V, 2)

    if proj_matrices.dim() == 3:
        proj_flat = proj_matrices.unsqueeze(0).expand(N * J, -1, -1, -1)
    else:
        proj_flat = (
            proj_matrices.unsqueeze(1)
            .expand(-1, J, -1, -1, -1)
            .reshape(N * J, V, 3, 4)
        )

    cov_flat = prec_flat = None
    if covariances is not None:
        cov_flat = covariances.permute(0, 2, 1, 3, 4).reshape(N * J, V, 2, 2)
    if precisions is not None:
        prec_flat = precisions.permute(0, 2, 1, 3, 4).reshape(N * J, V, 2, 2)

    conf_flat = None
    if confidences is not None:
        conf_flat = confidences.permute(0, 2, 1).reshape(N * J, V)

    X = triangulate_uncertainty_weighted(
        points_flat,
        proj_flat,
        covariances=cov_flat,
        precisions=prec_flat,
        confidences=conf_flat,
    )  # (N*J, 3)
    return X.view(N, J, 3)


class UncertaintyWeightedTriangulation(nn.Module):
    """Learnable uncertainty-weighted triangulation module.

    Predicts a per-view diagonal 2D covariance from the input 2D points using a
    small MLP, then triangulates with the predicted covariance as the
    uncertainty.  The module is differentiable and can be trained end-to-end.

    Parameters
    ----------
    hidden:
        Hidden dimension of the covariance-prediction MLP.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def _predict_covariances(self, points_2d: torch.Tensor) -> torch.Tensor:
        """Predict diagonal 2D covariances from 2D points.

        Args:
            points_2d: (B, V, 2)

        Returns:
            covariances: (B, V, 2, 2)
        """
        raw = self.mlp(points_2d)  # (B, V, 2)
        var = torch.nn.functional.softplus(raw) + 1e-4
        B, V, _ = var.shape
        covariances = torch.zeros(B, V, 2, 2, device=var.device, dtype=var.dtype)
        covariances[:, :, 0, 0] = var[:, :, 0]
        covariances[:, :, 1, 1] = var[:, :, 1]
        return covariances

    def forward(
        self,
        points_2d: torch.Tensor,
        proj_matrices: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            points_2d: (B, V, 2) for a single joint or (N, V, J, 2) for
                multiple joints.
            proj_matrices: (V, 3, 4), (B, V, 3, 4), or (N, V, 3, 4).
            confidences: optional scalar weights with shape (B, V) or (N, V, J).

        Returns:
            X: (B, 3) or (N, J, 3) triangulated 3D point(s).
        """
        if points_2d.dim() == 3:
            covariances = self._predict_covariances(points_2d)
            return triangulate_uncertainty_weighted(
                points_2d,
                proj_matrices,
                covariances=covariances,
                confidences=confidences,
            )

        if points_2d.dim() == 4:
            N, V, J, _ = points_2d.shape
            points_flat = points_2d.permute(0, 2, 1, 3).reshape(N * J, V, 2)
            cov_flat = self._predict_covariances(points_flat)

            if proj_matrices.dim() == 3:
                proj_flat = proj_matrices.unsqueeze(0).expand(N * J, -1, -1, -1)
            else:
                proj_flat = (
                    proj_matrices.unsqueeze(1)
                    .expand(-1, J, -1, -1, -1)
                    .reshape(N * J, V, 3, 4)
                )

            conf_flat = None
            if confidences is not None:
                conf_flat = confidences.permute(0, 2, 1).reshape(N * J, V)

            X = triangulate_uncertainty_weighted(
                points_flat,
                proj_flat,
                covariances=cov_flat,
                confidences=conf_flat,
            )
            return X.view(N, J, 3)

        raise ValueError(
            f"points_2d must be 3-D (B, V, 2) or 4-D (N, V, J, 2), "
            f"got shape {tuple(points_2d.shape)}"
        )


if __name__ == "__main__":
    import numpy as np

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
            cameras.append((K, R, t))
        return cameras

    rng = np.random.default_rng(42)
    X_true = np.array([1.0, 0.5, 3.0])
    cameras = _make_cameras(4)
    points_2d = []
    proj_matrices = []
    for K, R, t in cameras:
        P = K @ np.hstack([R, t[:, None]])
        x_h = P @ np.append(X_true, 1.0)
        x = x_h[:2] / x_h[2]
        points_2d.append(x)
        proj_matrices.append(P)

    points_2d = torch.from_numpy(np.stack(points_2d, axis=0)).float()
    proj_matrices = torch.from_numpy(np.stack(proj_matrices, axis=0)).float()

    X_recovered = triangulate_uncertainty_weighted(points_2d, proj_matrices)
    print("Recovered point:", X_recovered.detach().numpy())
    print("True point:", X_true)

    # Add noise to one view and downweight it.
    noisy_points = points_2d.clone()
    noisy_points[0] += 100.0
    covariances = torch.eye(2).unsqueeze(0).expand(4, -1, -1).clone()
    covariances[0] *= 1e6  # High covariance -> low weight.
    X_robust = triangulate_uncertainty_weighted(
        noisy_points, proj_matrices, covariances=covariances
    )
    print("Robust recovered point (noisy view 0 downweighted):", X_robust.detach().numpy())
