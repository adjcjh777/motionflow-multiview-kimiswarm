"""Bayesian triangulation v3: learned per-joint 3-D precision + refinement.

Extends ``RayAttentionFusionModelBayesianTriV2`` (batched-lstsq DLT,
view/joint anisotropic covariance, adaptive per-joint GN damping) with:

1. A per-joint 3-D precision head that predicts a Cholesky factor of a
   3x3 information matrix from pooled spatio-temporal features.
2. A learned refinement MLP that conditions each correction step on the
   per-joint precision parameters, producing a structured residual update.

The v3 head is intended as a drop-in prototype; it keeps the v2 triangulation
and epipolar loss untouched, only adding the extra precision/refinement branches
so it can be compared against v2 in a future full run.
"""

from typing import Tuple

import torch
import torch.nn as nn

from ..ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
    _adaptive_gauss_newton,
)
from ..triangulation import triangulate_dlt_batched_lstsq


def _params_to_cholesky_3d(params: torch.Tensor) -> torch.Tensor:
    """Build 3x3 lower-triangular Cholesky factors from 6 raw parameters.

    Args:
        params: (..., 6) ordered as [l00, l10, l11, l20, l21, l22].

    Returns:
        L: (..., 3, 3) lower-triangular matrix with positive diagonal.
    """
    shape = params.shape[:-1]
    L = torch.zeros(*shape, 3, 3, device=params.device, dtype=params.dtype)
    L[..., 0, 0] = torch.nn.functional.softplus(params[..., 0]) + 1e-4
    L[..., 1, 0] = params[..., 1]
    L[..., 1, 1] = torch.nn.functional.softplus(params[..., 2]) + 1e-4
    L[..., 2, 0] = params[..., 3]
    L[..., 2, 1] = params[..., 4]
    L[..., 2, 2] = torch.nn.functional.softplus(params[..., 5]) + 1e-4
    return L


class RayAttentionFusionModelBayesianTriV3(RayAttentionFusionModelBayesianTriV2):
    """Bayesian triangulation v3 with learned per-joint 3-D precision and refinement.

    Parameters
    ----------
    joint_precision_hidden:
        Hidden dimension of the per-joint 3-D precision head (default 64).
    refinement_hidden:
        Hidden dimension of the learned refinement head (default 128).
    n_refinement_iters:
        Number of iterative refinement steps (default 2).
    return_joint_precision:
        If True, return the predicted per-joint 3-D precision matrix.
    See ``RayAttentionFusionModelBayesianTriV2`` for the remaining arguments.
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
        joint_precision_hidden: int = 64,
        refinement_hidden: int = 128,
        n_refinement_iters: int = 2,
        return_joint_precision: bool = True,
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
            return_pp_delta=return_pp_delta,
            return_covariance=return_covariance,
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
        )
        self.return_joint_precision = return_joint_precision
        self.n_refinement_iters = max(1, n_refinement_iters)

        # Per-joint 3-D precision head: outputs 6 parameters for a 3x3 Cholesky factor.
        self.joint_precision_head = nn.Sequential(
            nn.Linear(d, joint_precision_hidden),
            nn.ReLU(),
            nn.Linear(joint_precision_hidden, 6),
        )

        # Learned refinement head that consumes feature, current 3-D estimate,
        # and the raw precision parameters.
        self.refinement_head = nn.Sequential(
            nn.Linear(d + 3 + 6, refinement_hidden),
            nn.ReLU(),
            nn.Linear(refinement_hidden, refinement_hidden),
            nn.ReLU(),
            nn.Linear(refinement_hidden, 3),
        )

    def _build_joint_precision(self, feat_pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict per-joint 3-D precision from pooled features.

        Args:
            feat_pooled: (N, J, d) pooled spatio-temporal features.

        Returns:
            L3d: (N, J, 3, 3) Cholesky factor of the information matrix.
            raw: (N, J, 6) raw precision parameters (kept for the refinement head).
        """
        raw = self.joint_precision_head(feat_pooled)  # (N, J, 6)
        L3d = _params_to_cholesky_3d(raw)
        return L3d, raw

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
            from ..ray_attention_temporal_crossview_model import _cameras_to_tensors
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

        visibility = self._visibility_multiplier(feat, confidences)

        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * precision * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt

        # V2: fully batched DLT.
        pred_3d_raw = triangulate_dlt_batched_lstsq(points_2d, P, weights)

        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        damping = self.damping_head(feat_pooled).squeeze(-1)
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

        # Per-joint 3-D precision from pooled features.
        L3d, raw_precision = self._build_joint_precision(feat_pooled)  # (B*T, J, 3, 3), (B*T, J, 6)

        # Learned iterative refinement conditioned on the per-joint precision.
        X = pred_3d_gn
        for _ in range(self.n_refinement_iters):
            refinement_input = torch.cat([feat_pooled, X, raw_precision], dim=-1)  # (B*T, J, d+3+6)
            delta = self.refinement_head(refinement_input)
            X = X + delta

        # Final residual MLP from pooled features and refined estimate.
        residual_input = torch.cat([feat_pooled, X], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = X + delta

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

        if self.return_joint_precision:
            L3d = L3d.view(B, T, J, 3, 3)
            if squeeze_output:
                L3d = L3d.squeeze(1)
            out += (L3d,)

        # Always append the auxiliary epipolar loss so the trainer can use it.
        out += (epi_loss,)
        return out


if __name__ == "__main__":
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

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
    model = RayAttentionFusionModelBayesianTriV3(
        j=J, d=64, n_views=V, gn_iters=2, epipolar_loss_weight=0.05
    )
    out = model(x, cameras=cameras)
    pred, weights, pp_delta, L3d, epi_loss = out
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)
    assert L3d.shape == (B, T, J, 3, 3)
    assert epi_loss.shape == ()
    loss = pred.mean() + 0.0 * epi_loss
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Bayesian triangulation v3 model smoke test passed")
