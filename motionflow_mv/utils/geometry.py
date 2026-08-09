"""Geometric utilities for multi-view pose estimation.

This module complements ``motionflow_mv/fusion/triangulation.py`` with
higher-level helpers that accept the batched ``(B, T, V, J, ...)`` tensors
used throughout ``OmniMultiViewFusionV5``.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_projection_matrix(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build projection matrices ``P = K [R | t]``.

    Args:
        K: (..., 3, 3) intrinsics.
        R: (..., 3, 3) rotations.
        t: (..., 3) translations.

    Returns:
        P: (..., 3, 4) projection matrices.
    """
    Rt = torch.cat([R, t[..., None]], dim=-1)  # (..., 3, 4)
    return torch.matmul(K, Rt)


def weighted_dlt_triangulate(
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    view_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    damping: float = 1e-4,
) -> torch.Tensor:
    """Batched weighted DLT triangulation with variable-view masking.

    Args:
        points_2d: (B, T, V, J, 2) 2-D keypoints.
        K: (B, T, V, 3, 3) camera intrinsics.
        R: (B, T, V, 3, 3) camera rotations.
        t: (B, T, V, 3) camera translations.
        weights: optional (B, T, V, J) positive weights. Defaults to 1.
        view_mask: optional (B, T, V) bool mask. True = view is valid.
        eps: small constant for numerical stability.
        damping: ridge regularisation on ``(A^T A)^{-1}``.

    Returns:
        X: (B, T, J, 3) triangulated 3-D points.
    """
    B, T, V, J, _ = points_2d.shape
    device = points_2d.device
    dtype = points_2d.dtype

    # Flatten (B, T) -> N so we can treat each frame independently.
    N = B * T
    pts = points_2d.reshape(N, V, J, 2)
    K_flat = K.reshape(N, V, 3, 3)
    R_flat = R.reshape(N, V, 3, 3)
    t_flat = t.reshape(N, V, 3)

    P = build_projection_matrix(K_flat, R_flat, t_flat)  # (N, V, 3, 4)

    if weights is None:
        weights_flat = torch.ones(N, V, J, device=device, dtype=dtype)
    else:
        weights_flat = weights.reshape(N, V, J)

    if view_mask is not None:
        mask = view_mask.reshape(N, V, 1).float()  # (N, V, 1)
        weights_flat = weights_flat * mask

    # Build the weighted DLT system for all joints/views.
    # Each view contributes two rows of shape (N, J, 4) which we stack into
    # (N, J, 2, 4) and then concatenate across views to (N, J, 2V, 4).
    A_rows = []
    for v in range(V):
        Pv = P[:, v, :, :]  # (N, 3, 4)
        x = pts[:, v, :, 0]  # (N, J)
        y = pts[:, v, :, 1]  # (N, J)

        row_x = x[..., None] * Pv[:, 2:3, :] - Pv[:, 0:1, :]  # (N, J, 4)
        row_y = y[..., None] * Pv[:, 2:3, :] - Pv[:, 1:2, :]  # (N, J, 4)

        w = weights_flat[:, v, :].unsqueeze(-1).sqrt().clamp(min=eps)  # (N, J, 1)
        A_view = torch.stack([row_x, row_y], dim=2) * w.unsqueeze(2)  # (N, J, 2, 4)
        A_rows.append(A_view)

    A = torch.cat(A_rows, dim=2)  # (N, J, 2V, 4)
    A3 = A[..., :3]  # (N, J, 2V, 3)
    b = -A[..., 3:]  # (N, J, 2V, 1)

    # Ridge-regularise to keep the least-squares systems full rank, especially
    # when only a few views are visible.
    if damping > 0.0:
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(N, J, -1, -1)
        A3 = torch.cat([A3, torch.sqrt(torch.tensor(damping, device=device, dtype=dtype)) * eye], dim=2)
        b = torch.cat([b, torch.zeros(N, J, 3, 1, device=device, dtype=dtype)], dim=2)

    # Batched least-squares over (N, J) independent 3x3 systems.
    X, *_ = torch.linalg.lstsq(A3, b)
    X = X.squeeze(-1)  # (N, J, 3)
    return X.reshape(B, T, J, 3)
