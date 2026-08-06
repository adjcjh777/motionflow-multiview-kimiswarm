"""Bayesian multi-view triangulation for the principal-point cross-view residual anchor.

Extends ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` with:

1. An anisotropic 2-D image-space covariance head that predicts a Cholesky factor
   per view / joint, yielding a learned precision matrix for weighted DLT.
2. An adaptive Gauss-Newton refinement step whose diagonal damping is predicted
   per joint from the pooled spatio-temporal features.
3. An epipolar-consistency auxiliary loss that uses the predicted covariances to
   weight pairwise fundamental-matrix constraints across views.

This is a paper-ablation style extension: the base principal-point correction,
cross-view spatio-temporal attention, and residual MLP are reused unchanged.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from .epipolar_attention_bias import compute_epipolar_distance


def _triangulate_weighted_dlt(points_2d, weights, P):
    """Local re-export of the DLT helper used by the ray-attention models."""
    from .ray_attention_model import _triangulate_weighted_dlt as _dlt
    return _dlt(points_2d, weights, P)


def _adaptive_gauss_newton(
    points_2d: torch.Tensor,
    weights: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    init_3d: torch.Tensor,
    damping: torch.Tensor,
    num_iters: int = 2,
) -> torch.Tensor:
    """Differentiable Gauss-Newton refinement with per-joint damping.

    Args:
        points_2d: (N, V, J, 2)
        weights:   (N, V, J) scalar weights for the residual terms.
        K, R, t:   (N, V, 3, 3), (N, V, 3, 3), (N, V, 3)
        init_3d:   (N, J, 3) initial estimate from DLT.
        damping:   (N, J) per-joint diagonal damping factor.
        num_iters: number of GN iterations.

    Returns:
        X: (N, J, 3) refined estimate.
    """
    N, V, J, _ = points_2d.shape
    X = init_3d

    fx = K[:, :, 0, 0]
    s = K[:, :, 0, 1]
    cx = K[:, :, 0, 2]
    fy = K[:, :, 1, 1]
    cy = K[:, :, 1, 2]
    eye3 = torch.eye(3, device=X.device, dtype=X.dtype).view(1, 1, 3, 3)

    for _ in range(max(1, num_iters)):
        X_cam = torch.einsum("nvab,njb->nvja", R, X) + t.unsqueeze(2)
        x_c, y_c, z_c = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
        inv_z = 1.0 / (z_c + 1e-8)

        u = (fx[:, :, None] * x_c + s[:, :, None] * y_c + cx[:, :, None] * z_c) * inv_z
        v = (fy[:, :, None] * y_c + cy[:, :, None] * z_c) * inv_z
        proj = torch.stack([u, v], dim=-1)
        r = points_2d - proj

        # Image Jacobian in camera frame.
        J_cam = torch.zeros(N, V, J, 2, 3, device=X.device, dtype=X.dtype)
        J_cam[:, :, :, 0, 0] = fx[:, :, None] * inv_z
        J_cam[:, :, :, 0, 1] = s[:, :, None] * inv_z
        J_cam[:, :, :, 0, 2] = (cx[:, :, None] - u) * inv_z
        J_cam[:, :, :, 1, 1] = fy[:, :, None] * inv_z
        J_cam[:, :, :, 1, 2] = (cy[:, :, None] - v) * inv_z

        # World-frame Jacobian.
        J_world = torch.einsum("nvjab,nvbd->nvjad", J_cam, R)
        J_world = J_world.permute(0, 2, 1, 3, 4).reshape(N, J, V * 2, 3)
        r_flat = r.permute(0, 2, 1, 3).reshape(N, J, V * 2)
        w_flat = (
            weights.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, 2)
            .reshape(N, J, V * 2)
        )

        A = torch.einsum("njkp,njkq->njpq", J_world, J_world * w_flat[..., None])
        b = torch.einsum("njkp,njk->njp", J_world, r_flat * w_flat)

        # Per-joint damping from the learned head.
        damp = damping.unsqueeze(-1).unsqueeze(-1)  # (N, J, 1, 1)
        A = A + damp * eye3.expand(N, J, -1, -1)

        b = b.unsqueeze(-1)
        dx = torch.linalg.solve(A, b).squeeze(-1)
        X = X + dx

    return X


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual model with anisotropic covariance and
    adaptive Gauss-Newton triangulation.

    Parameters
    ----------
    covariance_hidden:
        Hidden dimension of the Cholesky covariance head (default 64).
    gn_iters:
        Number of adaptive Gauss-Newton iterations (default 2).
    min_gn_damping, max_gn_damping:
        Range of the predicted per-joint GN damping.
    epipolar_loss_weight:
        Weight for the optional epipolar consistency term (forward returns it
        as the last element so the trainer can add it to the total loss).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_covariance: bool = False,
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=True,  # keep PP-delta branch for consistency
        )
        self.return_covariance = return_covariance
        self.gn_iters = gn_iters
        self.min_gn_damping = min_gn_damping
        self.max_gn_damping = max_gn_damping
        self.epipolar_loss_weight = epipolar_loss_weight

        # Cholesky covariance head: outputs l_xx, l_xy, l_yy for a 2x2
        # lower-triangular matrix.  Diagonal entries are softplus-ed to keep
        # the resulting covariance positive definite.
        self.covariance_head = nn.Sequential(
            nn.Linear(d, covariance_hidden),
            nn.ReLU(),
            nn.Linear(covariance_hidden, 3),
        )

        # Per-joint adaptive GN damping predictor.
        self.damping_head = nn.Sequential(
            nn.Linear(d, covariance_hidden),
            nn.ReLU(),
            nn.Linear(covariance_hidden, 1),
            nn.Sigmoid(),
        )

    def _cholesky_to_covariance(self, params: torch.Tensor) -> torch.Tensor:
        """Build 2x2 covariance matrices from the raw head output.

        Args:
            params: (..., 3) with raw [l_xx, l_xy, l_yy].

        Returns:
            L: (..., 2, 2) lower-triangular Cholesky factor.
        """
        l_xx = torch.nn.functional.softplus(params[..., 0]) + 1e-4
        l_xy = params[..., 1]
        l_yy = torch.nn.functional.softplus(params[..., 2]) + 1e-4

        L = torch.zeros(*params.shape[:-1], 2, 2, device=params.device, dtype=params.dtype)
        L[..., 0, 0] = l_xx
        L[..., 1, 0] = l_xy
        L[..., 1, 1] = l_yy
        return L

    def _epipolar_consistency_loss(
        self,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        L: torch.Tensor,
    ) -> torch.Tensor:
        """Geometry-regularized consistency weighted by predicted covariance.

        Uses the symmetric epipolar distance from ``compute_epipolar_distance``
        and weights each view-pair contribution by the harmonic mean of the
        predicted determinants.
        """
        # (N, V_src, V_dst, J)
        dist = compute_epipolar_distance(K, R, t, points_2d)  # noqa: F841
        # Determinant of each 2x2 covariance.
        det = (L[..., 0, 0] * L[..., 1, 1]) ** 2  # (N, V, J)
        # Pairwise precision weight: 1 / (det_src + det_dst + eps)
        det_src = det.unsqueeze(2)  # (N, V, 1, J)
        det_dst = det.unsqueeze(1)  # (N, 1, V, J)
        pair_weight = 1.0 / (det_src + det_dst + 1e-6)
        # The loss encourages small epipolar distance where precision is high.
        loss = (pair_weight * dist).mean()
        return loss

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Anisotropic covariance prediction per (view, joint).
        raw_cov = self.covariance_head(feat)  # (B*T, V, J, 3)
        L = self._cholesky_to_covariance(raw_cov)  # (B*T, V, J, 2, 2)

        # Precision weight = 1 / sqrt(det(Σ)) = 1 / (l_xx * l_yy).
        precision = 1.0 / (
            L[..., 0, 0].clamp(min=1e-4) * L[..., 1, 1].clamp(min=1e-4)
        )

        # Optional visibility-aware weighting from base class.
        visibility = self._visibility_multiplier(feat, confidences)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * precision * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Adaptive Gauss-Newton refinement.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        damping = self.damping_head(feat_pooled).squeeze(-1)  # (B*T, J)
        damping = self.min_gn_damping + (
            self.max_gn_damping - self.min_gn_damping
        ) * damping

        pred_3d_gn = _adaptive_gauss_newton(
            points_2d,
            weights,
            K_corrected,
            R,
            t,
            pred_3d_raw,
            damping,
            num_iters=self.gn_iters,
        )

        # Residual refinement head.
        residual_input = torch.cat([feat_pooled, pred_3d_gn], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_gn + delta

        # Epipolar consistency loss (used as auxiliary in the trainer).
        epi_loss = self._epipolar_consistency_loss(points_2d, K_corrected, R, t, L)
        epi_loss = self.epipolar_loss_weight * epi_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        L = L.view(B, T, V, J, 2, 2)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            L = L.squeeze(1)

        # Build output tuple consistent with the base model, appending the new
        # diagnostics at the end.
        out = (pred_3d, weights)

        if self.return_pp_delta:
            out += (pp_delta,)
            if self.correct_focal:
                out += (focal_scale,)

        if self.return_covariance:
            out += (L,)

        # Always append the auxiliary epipolar loss so the trainer can use it.
        out += (epi_loss,)
        return out


if __name__ == "__main__":
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
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointBayesianTri(
        j=J, d=64, n_views=V, gn_iters=2, epipolar_loss_weight=0.05
    )
    pred, weights, pp_delta, epi_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert epi_loss.shape == ()
    loss = pred.mean() + 0.0 * epi_loss  # ensure graph connection
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Bayesian triangulation model smoke test passed")
