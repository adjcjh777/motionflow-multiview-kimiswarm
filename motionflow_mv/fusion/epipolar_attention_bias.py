"""Epipolar-line distance utilities for calibrated multi-view attention.

These functions are pure geometry and fully differentiable w.r.t. the camera
parameters.  They compute the image-space distance from a 2-D keypoint in one
view to the epipolar line induced by the corresponding keypoint in another view.
"""

import torch


def _skew_symmetric(v):
    """Return the skew-symmetric matrix of a vector."""
    b = v.shape[0]
    zero = torch.zeros(b, device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1),
    ], dim=1)  # (B, 3, 3)


def compute_epipolar_distance(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    points_2d: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-joint epipolar-line distances for every view pair.

    Args
    ----
    K: ``(B, V, 3, 3)`` intrinsic matrices.
    R: ``(B, V, 3, 3)`` rotation matrices.
    t: ``(B, V, 3)`` translation vectors.
    points_2d: ``(B, V, J, 2)`` image points.
    eps: small constant.

    Returns
    -------
    dist: ``(B, V_src, V_dst, J)`` unsigned distance from each joint in the
    destination view to the epipolar line induced by the same joint in the
    source view.
    """
    B, V, _ = K.shape[:3]
    device = K.device
    dtype = K.dtype

    # Sanity check: inputs must be finite and intrinsics must be invertible.
    # Mixed datasets with padded views can pass identity/zero-translation cameras
    # that are technically invertible but geometrically degenerate; callers are
    # responsible for masking padded views before calling this function.
    if not (torch.isfinite(K).all() and torch.isfinite(R).all() and torch.isfinite(t).all()):
        raise ValueError("Non-finite values in camera parameters (K, R, t).")
    if not (torch.isfinite(points_2d).all()):
        raise ValueError("Non-finite values in 2D points.")
    det_K = torch.det(K)
    if (det_K.abs() < eps).any():
        bad = (det_K.abs() < eps).nonzero(as_tuple=True)
        raise RuntimeError(
            f"Singular intrinsic matrices detected for batch/view indices {bad} "
            f"(abs(det) < {eps}). Check camera calibration and view masking."
        )

    # Inverse intrinsics.
    K_inv = torch.inverse(K)  # (B, V, 3, 3)

    # For every src/dst pair compute F_{dst,src}.
    # R_rel[v_src, v_dst] = R_dst R_src^T  (world rotation from src to dst)
    # t_rel[v_src, v_dst] = t_dst - R_rel t_src
    R_src = R.unsqueeze(1)            # (B, 1, V, 3, 3)
    R_dst = R.unsqueeze(2)            # (B, V, 1, 3, 3)
    t_src = t.unsqueeze(1)            # (B, 1, V, 3)
    t_dst = t.unsqueeze(2)            # (B, V, 1, 3)

    R_rel = R_dst @ R_src.transpose(-2, -1)  # (B, V, V, 3, 3)
    t_rel = t_dst - (R_rel @ t_src.unsqueeze(-1)).squeeze(-1)  # (B, V, V, 3)

    # Essential matrix E = [t_rel]_x R_rel, Fundamental F = K_dst^{-T} E K_src^{-1}
    t_rel_flat = t_rel.reshape(B * V * V, 3)
    skew = _skew_symmetric(t_rel_flat).reshape(B, V, V, 3, 3)
    E = skew @ R_rel  # (B, V, V, 3, 3)

    K_inv_src = K_inv.unsqueeze(1)  # (B, 1, V, 3, 3)
    K_inv_dst = K_inv.unsqueeze(2)  # (B, V, 1, 3, 3)
    F = K_inv_dst.transpose(-2, -1) @ E @ K_inv_src  # (B, V, V, 3, 3)

    # Build homogeneous points (x, y, 1).
    ones = torch.ones(B, V, points_2d.shape[2], 1, device=device, dtype=dtype)
    pts = torch.cat([points_2d, ones], dim=-1)  # (B, V, J, 3)

    # Epipolar line in dst: l = F_src_dst @ x_src.
    # F indexed as (B, V_src, V_dst, 3, 3); we need line in dst for src->dst.
    # For a given src i and dst j, l_j = F[j, i] @ x_i.
    # pts shape (B, V, J, 3); rearrange to (B, V_src, 1, J, 3) for matmul.
    F_mat = F.permute(0, 2, 1, 3, 4)  # (B, V_dst, V_src, 3, 3) -> easier to loop? use einsum

    # Use einsum: F_mat[b, j, i, :, :] * pts[b, i, k, :] -> l[b, j, i, k, :]
    # F (B, V_src, V_dst, 3, 3). For each src i, dst j.
    # F_exp[b,i,j,3,3] @ pts_src[b,i,J,3]^T -> l[b,i,j,J,3]
    l = torch.einsum("bijmn,bikn->bijkm", F, pts)  # (B, V_src, V_dst, J, 3)

    # Distance from point in dst to line l.
    numerator = torch.abs(torch.einsum("bijkm,bjkm->bijk", l, pts))  # (B, V_src, V_dst, J)
    denom = torch.sqrt(l[..., 0] ** 2 + l[..., 1] ** 2 + eps)  # (B, V_src, V_dst, J)
    dist = numerator / denom
    return dist


def epipolar_bias_from_distance(
    dist: torch.Tensor,
    temperature: float = 100.0,
    clip_max: float = None,
) -> torch.Tensor:
    """Convert epipolar distances to an additive attention/weight bias.

    Args
    ----
    dist: ``(B, V_src, V_dst, J)`` distances.
    temperature: scaling factor (pixels).
    clip_max: optional upper clip on the bias magnitude.

    Returns
    -------
    bias: ``(B, V_dst, J)`` aggregated bias suitable for weight logits of shape
    ``(B, J, V_dst)``.
    """
    if clip_max is not None:
        dist = dist.clamp(max=clip_max)
    # Lower distance -> higher confidence in this view pair -> positive bias for dst.
    # Aggregate over source views: average penalty; we subtract it from logits.
    bias = -(dist.mean(dim=1)) / temperature  # (B, V_dst, J)
    return bias
