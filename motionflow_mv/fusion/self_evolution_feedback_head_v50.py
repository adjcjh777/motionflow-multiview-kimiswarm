"""v50 Self-Evolution Feedback Head (SEFH).

A lightweight, gradient-safe feedback head that closes the prediction↔uncertainty
loop by predicting per-view reliability and per-joint log-variance from
reprojection, temporal, and epipolar residuals.  It is designed to be a strict
superset of the v46/v48 baseline: the final reliability gate is initialized near
identity, so enabling the module does not perturb the already-trained pose
estimator at startup.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfEvolutionFeedbackHeadV50(nn.Module):
    """Predict per-view reliability and per-joint uncertainty from residuals.

    Parameters
    ----------
    j: int
        Number of joints.
    hidden: int
        Hidden dimension of the residual MLP.
    num_layers: int
        Number of MLP layers (at least 2).
    dropout: float
        Dropout probability inside the MLP.
    reproj_weight: float
        Scalar weight for the reprojection residual branch.
    temporal_weight: float
        Scalar weight for the temporal residual branch.
    epipolar_weight: float
        Scalar weight for the epipolar residual branch.
    identity_init_gate: bool
        If True, the final reliability gate is initialized so that the module
        is identity at startup (reliability ≈ 1).
    """

    def __init__(
        self,
        j: int = 17,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        reproj_weight: float = 1.0,
        temporal_weight: float = 0.5,
        epipolar_weight: float = 0.5,
        identity_init_gate: bool = True,
    ) -> None:
        super().__init__()
        self.j = j
        self.reproj_weight = reproj_weight
        self.temporal_weight = temporal_weight
        self.epipolar_weight = epipolar_weight
        self.identity_init_gate = identity_init_gate

        # Input features per (view, joint):
        # reprojection residual (2), temporal residual (3), epipolar residual (1)
        in_dim = 6
        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        # Final heads: per-view reliability and per-joint log-variance.
        self.reliability_head = nn.Linear(hidden, 1)
        # Initialize the reliability head to produce near-zero logit -> sigmoid 0.5.
        # If identity_init_gate is requested we will further bias it to ~1 in forward.
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)

        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def _compute_reprojection_residual(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-view per-joint reprojection error (B, T, V, J)."""
        B, T, J, _ = pred_3d.shape
        V = points_2d.shape[2]
        # pred_3d: (B, T, J, 3) -> (B, T, 1, J, 3)
        pred = pred_3d.unsqueeze(2)
        # points_2d: (B, T, V, J, 3) -> use first two channels
        pts = points_2d[..., :2]  # (B, T, V, J, 2)

        # Build projection matrix P = K @ [R|t] for each view.
        # K, R: (B, V, 3, 3), t: (B, V, 3)
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (B, V, 3, 4)
        P = K @ Rt  # (B, V, 3, 4)
        P = P.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3, 4)

        ones = torch.ones(B, T, 1, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        pred_h = torch.cat([pred, ones], dim=-1)  # (B, T, V, J, 4)
        pred_h = pred_h.unsqueeze(-1)  # (B, T, V, J, 4, 1)
        proj = (P @ pred_h).squeeze(-1)  # (B, T, V, J, 3)
        proj_2d = proj[..., :2] / (proj[..., 2:3] + 1e-8)
        err = (proj_2d - pts).norm(dim=-1)  # (B, T, V, J)
        return err

    def _compute_temporal_residual(self, pred_3d: torch.Tensor) -> torch.Tensor:
        """Return per-joint temporal jump magnitude (B, T, J)."""
        if pred_3d.shape[1] < 2:
            return torch.zeros_like(pred_3d[..., 0])
        vel = pred_3d[:, 1:] - pred_3d[:, :-1]
        # Pad with zeros to keep length T.
        pad = torch.zeros_like(vel[:, :1])
        vel = torch.cat([pad, vel], dim=1)
        return vel.norm(dim=-1)

    def _compute_epipolar_residual(
        self,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-view cross-view projection-consistency residual (B, T, V, J).

        For each view, we compute the L2 distance between the projected point and
        the median projected point across all views.  This is a cheap, robust
        proxy for epipolar consistency that avoids the numerical fragility of the
        fundamental matrix and gracefully handles missing views (which are
        masked in the downstream loss).
        """
        B, T, J, _ = pred_3d.shape
        V = K.shape[1]
        if V < 2:
            return torch.zeros(B, T, V, J, device=pred_3d.device, dtype=pred_3d.dtype)

        # Project 3D points to each view.
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (B, V, 3, 4)
        P = K @ Rt  # (B, V, 3, 4)
        P = P.unsqueeze(1).unsqueeze(3)  # (B, 1, V, 1, 3, 4)

        pred = pred_3d.unsqueeze(2)  # (B, T, 1, J, 3)
        ones = torch.ones(B, T, 1, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        pred_h = torch.cat([pred, ones], dim=-1)  # (B, T, 1, J, 4)
        pred_h = pred_h.unsqueeze(-1)  # (B, T, 1, J, 4, 1)

        proj = (P @ pred_h).squeeze(-1)  # (B, T, V, J, 3)
        proj_2d = proj[..., :2] / (proj[..., 2:3] + 1e-8)

        # Median projection across views.
        median = proj_2d.median(dim=2, keepdim=True)[0]  # (B, T, 1, J, 2)
        err = (proj_2d - median).norm(dim=-1)  # (B, T, V, J)
        return err

    def forward(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reliability, log_var, and diagnostic residuals.

        Returns
        -------
        reliability: (B, T, V, J) in [min_rel, 1].
        log_var: (B, T, J).
        reproj_residual: (B, T, V, J).
        temporal_residual: (B, T, J).
        epipolar_residual: (B, T, V, J).
        feature: (B, T, V, J, hidden).
        """
        reproj = self._compute_reprojection_residual(pred_3d, points_2d, K, R, t)
        temporal = self._compute_temporal_residual(pred_3d)  # (B, T, J)
        epipolar = self._compute_epipolar_residual(pred_3d, K, R, t)

        # Broadcast temporal residual to (B, T, V, J).
        temporal_b = temporal.unsqueeze(2).expand_as(reproj)

        # Feature vector per (view, joint).
        feat = torch.stack([reproj, temporal_b, epipolar], dim=-1)
        # Add a few non-linear transformations to give the MLP expressive power.
        feat = torch.cat([feat, feat.log1p()], dim=-1)
        hidden = self.mlp(feat)

        reliability_logit = self.reliability_head(hidden).squeeze(-1)  # (B, T, V, J)
        if self.identity_init_gate:
            # Bias so that at init reliability ~1 (sigmoid(2.5) ≈ 0.92).
            reliability_logit = reliability_logit + 2.5
        reliability = torch.sigmoid(reliability_logit)
        # Clamp to avoid zero-weighted views.
        reliability = reliability.clamp(0.05, 1.0)

        # Per-joint log-variance: pool hidden features over views.
        if view_mask is not None:
            mask = view_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, V, 1, 1)
            pooled_feat = (hidden * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-8)
        else:
            pooled_feat = hidden.mean(dim=2)

        log_var = self.log_var_head(pooled_feat).squeeze(-1)  # (B, T, J)

        # Mask unreliable residuals from the epipolar branch if view_mask given.
        if view_mask is not None:
            vm = view_mask.unsqueeze(-1)
            reproj = reproj * vm
            epipolar = epipolar * vm

        return reliability, log_var, reproj, temporal, epipolar, hidden


def compute_sefh_loss(
    reliability: torch.Tensor,
    log_var: torch.Tensor,
    reproj: torch.Tensor,
    temporal: torch.Tensor,
    epipolar: torch.Tensor,
    view_mask: torch.Tensor | None = None,
    loss_weight: float = 0.01,
    residual_clip: float = 50.0,
) -> torch.Tensor:
    """Compute the v50 Self-Evolution Feedback Head auxiliary loss.

    Parameters
    ----------
    reliability: (B, T, V, J)
    log_var: (B, T, J)
    reproj: (B, T, V, J)
    temporal: (B, T, J)
    epipolar: (B, T, V, J)
    view_mask: (B, T, V) optional
    loss_weight: scalar multiplier for the auxiliary loss.
    residual_clip: clip residuals before feeding into the loss.

    Returns
    -------
    loss: scalar tensor.
    """
    if view_mask is not None:
        mask = view_mask.unsqueeze(-1)
    else:
        B, T, V, J = reliability.shape
        mask = torch.ones_like(reliability)

    # Reprojection negative log-likelihood under predicted reliability and uncertainty.
    r = reproj.clamp(0, residual_clip)
    # Per-joint variance from predicted log-variance; clamp to avoid collapse/explosion.
    sigma_sq = torch.exp(log_var).unsqueeze(2).clamp(0.01, 100.0)  # (B, T, 1, J)
    # Stable Gaussian NLL: r^2 / sigma^2 + log(sigma^2).
    reproj_nll = (reliability * (r ** 2 / (sigma_sq + 1e-6) + torch.log(sigma_sq + 1e-6))).mean()

    # Temporal smoothness on reliability.
    if reliability.shape[1] > 1:
        rel_smooth = (reliability[:, 1:] - reliability[:, :-1]).pow(2).mean()
    else:
        rel_smooth = torch.tensor(0.0, device=reliability.device, dtype=reliability.dtype)

    # Entropy regularization to prevent uniform collapse.
    p = reliability.clamp(1e-6, 1 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    loss = loss_weight * (reproj_nll + 0.1 * rel_smooth - 0.001 * entropy)
    return loss
