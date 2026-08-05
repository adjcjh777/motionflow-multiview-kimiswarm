"""Temporal ray-aware attention fusion with uncertainty-weighted DLT (v2).

Extends ``RayAttentionFusionModelTemporalResidualV2`` by replacing the sigmoid
per-view weight head with a Gaussian uncertainty head.  For each view and joint
the model predicts a log-variance ``log_var``; the weighted DLT then uses

    weight = confidence * exp(-log_var)

so that views predicted to be uncertain receive lower weight.  An optional
per-view reprojection NLL auxiliary loss supervises the predicted uncertainties.

The v2 variant keeps the V4-normalised camera embedding from
``RayAttentionFusionModelTemporalResidualV2``, so it is bit-for-bit compatible
with V4 checkpoints and should inherit their better cross-scale generalisation.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_temporal_residual_v2_model import RayAttentionFusionModelTemporalResidualV2
from .ray_attention_model import _triangulate_weighted_dlt
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class RayAttentionFusionModelTemporalUncertaintyV2(RayAttentionFusionModelTemporalResidualV2):
    """Temporal ray-attention fusion with uncertainty-weighted triangulation (v2).

    Input / output semantics are identical to ``RayAttentionFusionModelTemporalResidual``
    plus an extra ``log_var`` tensor and ``nll_loss`` scalar for the uncertainty term.

    Parameters
    ----------
    residual_hidden:
        Hidden dimension of the residual MLP (default 128).
    uncertainty_loss_weight:
        Weight for the reprojection NLL auxiliary loss (default 0.1).
    log_var_min, log_var_max:
        Clamping range for the predicted log variance.
    **kwargs:
        Passed to ``RayAttentionFusionModelTemporalResidualV2``.
    """

    def __init__(
        self,
        residual_hidden: int = 128,
        uncertainty_loss_weight: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
        **kwargs,
    ):
        # Pass residual_hidden up; keep use_reproj_gate as a kwarg if provided.
        super().__init__(residual_hidden=residual_hidden, **kwargs)
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Predict a scalar log-variance per (view, joint) from the temporal token.
        self.uncertainty_head = nn.Linear(self.d, 1)

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood.

        Args:
            points_2d: (N, V, J, 2)
            pred_3d: (N, J, 3)
            proj_matrices: (N, V, 3, 4)
            log_var: (N, V, J)

        Returns:
            nll: scalar
        """
        N, V, J, _ = points_2d.shape
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)  # (N, J, 4)
        p_h = torch.einsum("nvij,nkj->nvki", proj_matrices, Xh)
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z  # (N, V, J, 2)
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)  # (N, V, J)
        # Gaussian NLL up to constants: 0.5 * (err^2 / var + log_var)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            K, R, t = _cameras_to_tensors(cameras, device)

        # Prepare per-sample camera tensors and flatten time into batch for the
        # per-frame encoder.
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

        # Per-frame v4 features (inherited from V2 encoder).
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Per-frame uncertainty prediction and DLT triangulation.
        feat_for_uncertainty = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)  # (B*T, J, V)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)  # (B*T, V, J)

        # Use precision as the DLT weight, scaled by observed confidence.
        precision = torch.exp(-log_var)
        weights = precision * confidences  # (B*T, V, J)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head (inherited from V2 / residual model).
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        pred_3d = pred_3d_raw
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)  # (B*T, J, d+3)
            delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
            if self.use_reproj_gate:
                summary = self._reprojection_error_summary(
                    pred_3d, points_2d, P, inlier_thresh=10.0
                )
                gate_input = torch.cat([residual_input, summary], dim=-1)
                gate = self.reproj_gate(gate_input)  # (B*T, J, 1)
                delta = gate * delta
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


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    from ..calibration.camera import Camera
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


if __name__ == "__main__":
    # Quick shape/gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalUncertaintyV2(j=J, d=64, n_views=V)
    pred, w, log_var, nll = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    assert log_var.shape == (B, T, V, J)
    loss = pred.mean() + nll
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("temporal uncertainty v2 model sanity check passed")

    # Iterative refinement sanity check.
    pred_iter, _, _, _ = model(x, cameras=cameras, n_iter=3)
    assert pred_iter.shape == (B, T, J, 3)
    print("iterative refinement sanity check passed")
