"""Temporal ray-aware attention fusion with uncertainty, residual and learned triangulation.

Extends ``RayAttentionFusionModelTemporalResidual`` by adding an uncertainty
head and a differentiable Gauss-Newton triangulation refinement step, but keeps
the temporal-only (no cross-view) attention.  This tests whether the learned
triangulation/uncertainty components help without the extra cross-view
complexity.

Input / output semantics are identical to ``RayAttentionFusionModelTemporalResidual``
plus an extra ``nll_loss`` returned for the uncertainty term.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .ray_attention_model import _triangulate_weighted_dlt


class RayAttentionFusionModelTemporalUncertaintyResidualLearnedTriV1(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-aware fusion with uncertainty, residual and learned triangulation.

    Parameters
    ----------
    gn_iters:
        Number of Gauss-Newton iterations (default 3).  Set to 0 to skip GN.
    gn_damping:
        Diagonal damping for the GN normal equations.
    uncertainty_loss_weight:
        Weight for the reprojection NLL auxiliary loss.
    **kwargs:
        Passed to ``RayAttentionFusionModelTemporalResidual``.
    """

    def __init__(
        self,
        gn_iters: int = 3,
        gn_damping: float = 1e-6,
        uncertainty_loss_weight: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gn_iters = gn_iters
        self.gn_damping = gn_damping
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        d = self.d
        self.uncertainty_head = nn.Linear(d, 1)

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood."""
        N, V, J, _ = points_2d.shape
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)
        p_h = torch.einsum('nvij,nkj->nvki', proj_matrices, Xh)
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

    def _triangulate_weighted_gauss_newton(
        self,
        points_2d: torch.Tensor,
        weights: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        init_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable weighted Gauss-Newton triangulation."""
        N, V, J, _ = points_2d.shape
        X = init_3d

        fx = K[:, :, 0, 0]
        s = K[:, :, 0, 1]
        cx = K[:, :, 0, 2]
        fy = K[:, :, 1, 1]
        cy = K[:, :, 1, 2]
        eye3 = torch.eye(3, device=X.device, dtype=X.dtype).view(1, 1, 3, 3)

        for _ in range(max(1, self.gn_iters)):
            X_cam = torch.einsum("nvab,njb->nvja", R, X) + t.unsqueeze(2)
            x_c, y_c, z_c = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
            inv_z = 1.0 / (z_c + 1e-8)
            u = (fx[:, :, None] * x_c + s[:, :, None] * y_c + cx[:, :, None] * z_c) * inv_z
            v = (fy[:, :, None] * y_c + cy[:, :, None] * z_c) * inv_z
            proj = torch.stack([u, v], dim=-1)
            r = points_2d - proj

            J_cam = torch.zeros(N, V, J, 2, 3, device=X.device, dtype=X.dtype)
            J_cam[:, :, :, 0, 0] = fx[:, :, None] * inv_z
            J_cam[:, :, :, 0, 1] = s[:, :, None] * inv_z
            J_cam[:, :, :, 0, 2] = (cx[:, :, None] - u) * inv_z
            J_cam[:, :, :, 1, 1] = fy[:, :, None] * inv_z
            J_cam[:, :, :, 1, 2] = (cy[:, :, None] - v) * inv_z

            J_world = torch.einsum("nvjab,nvbd->nvjad", J_cam, R)
            J_world = J_world.permute(0, 2, 1, 3, 4).reshape(N, J, V * 2, 3)
            r_flat = r.permute(0, 2, 1, 3).reshape(N, J, V * 2)
            w_flat = weights.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 2).reshape(N, J, V * 2)

            A = torch.einsum("njkp,njkq->njpq", J_world, J_world * w_flat[..., None])
            b = torch.einsum("njkp,njk->njp", J_world, r_flat * w_flat)

            A = A + self.gn_damping * eye3.expand(N, J, -1, -1)
            b = b.unsqueeze(-1)
            dx = torch.linalg.solve(A, b).squeeze(-1)
            X = X + dx

        return X

    def forward(self, x, cameras=None, K=None, R=None, t=None, n_iter: int = 1):
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

        feat = self._extract_frame_features(x_flat, K, R, t)

        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        feat_for_uncertainty = feat.permute(0, 2, 1, 3)
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)

        precision = torch.exp(-log_var)
        weights = precision * confidences
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        if self.gn_iters > 0:
            pred_3d_gn = self._triangulate_weighted_gauss_newton(points_2d, weights, K, R, t, pred_3d_raw)
        else:
            pred_3d_gn = pred_3d_raw

        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_gn
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d + delta

        nll_loss = self._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = self.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        log_var = log_var.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            log_var = log_var.squeeze(1)

        return pred_3d, weights, log_var, nll_loss
