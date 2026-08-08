"""Uncertainty-aware depth-proposal triangulation (v27).

Replaces the fixed uniform depth grid in ``DepthProposalTriangulation`` with a
learned per-ray Gaussian depth distribution.  Optionally supports a Gaussian
mixture model (``n_mixtures > 1``) over depth along each ray.  The head predicts
mixture weights, means and log-standard-deviations, samples depth hypotheses
continuously, and scores them with an MLP that also sees the predicted
uncertainty.  At inference the deterministic mixture mean is used, so the head
remains fast.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyDepthProposalTriangulation(nn.Module):
    """Learned per-ray depth distribution for multi-view triangulation.

    Parameters
    ----------
    n_views:
        Number of views (shape hint).
    n_ray_samples:
        Number of Monte-Carlo depth samples drawn per ray during training.
    n_mixtures:
        Number of Gaussian mixture components for the per-ray depth distribution.
    init_z_min, init_z_max:
        Initial depth range.  Component means are initialised across this range.
    min_sigma:
        Floor on the predicted standard deviation to avoid collapse.
    uncertainty_loss_weight:
        Weight of the uncertainty regularisation term added to the geometry loss.
    """

    def __init__(
        self,
        n_views: int,
        n_ray_samples: int = 4,
        n_mixtures: int = 1,
        init_z_min: float = 1.0,
        init_z_max: float = 8.0,
        min_sigma: float = 0.05,
        uncertainty_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.n_views = n_views
        self.n_ray_samples = n_ray_samples
        self.n_mixtures = max(1, n_mixtures)
        self.init_z_min = init_z_min
        self.init_z_max = init_z_max
        self.min_sigma = min_sigma
        self.uncertainty_loss_weight = uncertainty_loss_weight

        # Per-ray depth distribution parameters.
        # Input: ray origin, ray direction, current 3D estimate, confidence.
        # Output per mixture component: logit_weight, mu, log_sigma.
        self.dist_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * self.n_mixtures),
        )
        # Initialise to a reasonable depth range.  When n_mixtures>1, spread the
        # component means across the range; otherwise place a single component in
        # the middle.  All weights start uniform (logits 0) and sigmas start at 1.
        nn.init.zeros_(self.dist_mlp[-1].weight)
        with torch.no_grad():
            out_bias = self.dist_mlp[-1].bias.view(self.n_mixtures, 3)
            out_bias[:, 0] = 0.0
            if self.n_mixtures == 1:
                out_bias[:, 1] = (init_z_min + init_z_max) / 2.0
            else:
                out_bias[:, 1] = torch.linspace(init_z_min, init_z_max, self.n_mixtures)
            out_bias[:, 2] = 0.0

        # Score each (view, sample) candidate.  Feat: candidate + current estimate + uncertainty.
        self.score_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Zero final layer -> uniform scores at init.
        for p in self.score_mlp[-1].parameters():
            nn.init.zeros_(p)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        centre: torch.Tensor,
        direction: torch.Tensor,
        confidence: torch.Tensor,
        pred_3d: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            centre: (B, T, V, 3) ray origins (camera centres).
            direction: (B, T, V, J, 3) unit ray directions.
            confidence: (B, T, V, J) per-view confidence.
            pred_3d: (B, T, J, 3) current 3D estimate.
            view_mask: optional (B, T, V) bool mask.

        Returns:
            refined: (B, T, J, 3) refined 3D joints.
            uncertainty_loss: scalar regularisation term.
        """
        B, T, V, J = direction.shape[:4]

        # Per-ray depth distribution.
        # Expand pred_3d to each view.
        pred_exp = pred_3d[:, :, None, :, :].expand(-1, -1, V, -1, -1)  # (B, T, V, J, 3)
        conf_exp = confidence[..., None]  # (B, T, V, J, 1)
        centre_exp = centre[:, :, :, None, :].expand(-1, -1, -1, J, -1)  # (B, T, V, J, 3)
        dist_feat = torch.cat([centre_exp, direction, pred_exp, conf_exp], dim=-1)  # (B, T, V, J, 10)
        dist_out = self.dist_mlp(dist_feat)  # (B, T, V, J, 3*K)
        dist_out = dist_out.view(B, T, V, J, self.n_mixtures, 3)
        mix_logits = dist_out[..., 0]  # (B, T, V, J, K)
        mu = dist_out[..., 1]
        log_sigma = dist_out[..., 2]
        mix_weights = F.softmax(mix_logits, dim=-1)  # (B, T, V, J, K)
        sigma = torch.exp(log_sigma).clamp(min=self.min_sigma)

        # Sample depths: during training reparameterised, at inference use mean.
        if self.training:
            # Standard Normal samples for each mixture component.
            eps = torch.randn(B, T, V, J, self.n_mixtures, self.n_ray_samples, device=direction.device, dtype=direction.dtype)
            # Clamp sigma for stability.
            sigma_sample = sigma[..., None].clamp(max=2.0)  # (B, T, V, J, K, 1)
            # Per-component samples.
            z_components = mu[..., None] + sigma_sample * eps  # (B, T, V, J, K, S)
            # Soft mixture: weighted sum over components.  This is differentiable
            # in both the means and the mixture weights.
            mix_weights_exp = mix_weights[..., None]  # (B, T, V, J, K, 1)
            z_vals = (mix_weights_exp * z_components).sum(dim=4)  # (B, T, V, J, S)
        else:
            # Deterministic mixture mean.
            z_mean = (mix_weights * mu).sum(dim=-1)  # (B, T, V, J)
            z_vals = z_mean[..., None].expand(-1, -1, -1, -1, self.n_ray_samples)

        # Candidate points: c_v + z * d_vj.
        candidates = (
            centre_exp[..., None, :]
            + z_vals[..., None] * direction[..., None, :]
        )  # (B, T, V, J, S, 3)

        # Score candidates with uncertainty-aware MLP.
        # Features: candidate, current estimate (broadcast), predicted sigma.
        # Use the mixture-weighted std for the sigma feature.
        sigma_weighted = (mix_weights * sigma).sum(dim=-1)  # (B, T, V, J)
        sigma_exp = sigma_weighted[..., None].expand(-1, -1, -1, -1, self.n_ray_samples)
        pred_exp_s = pred_3d[:, :, None, :, None, :].expand(-1, -1, V, -1, self.n_ray_samples, -1)
        score_feat = torch.cat([candidates, pred_exp_s, sigma_exp[..., None]], dim=-1)
        scores = self.score_mlp(score_feat).squeeze(-1)  # (B, T, V, J, S)

        if view_mask is not None:
            scores = scores.masked_fill(~view_mask[:, :, :, None, None], float("-inf"))

        # Softmax over the (V, S) candidate dimension.
        scores_flat = scores.view(B, T, V * self.n_ray_samples, J).permute(0, 1, 3, 2)  # (B, T, J, V*S)
        scores_flat = torch.where(
            torch.isinf(scores_flat).all(dim=-1, keepdim=True),
            torch.zeros_like(scores_flat),
            scores_flat,
        )
        probs = F.softmax(scores_flat, dim=-1)
        candidates_flat = candidates.view(B, T, V * self.n_ray_samples, J, 3).permute(0, 1, 3, 2, 4)
        fused = (probs[..., None] * candidates_flat).sum(dim=3)  # (B, T, J, 3)

        # Identity-at-init residual.
        residual = self.fusion_mlp(fused - pred_3d)
        refined = pred_3d + self.residual_scale * residual

        # Uncertainty regularisation: keep the weighted std close to a target.
        sigma_target = 0.2
        uncertainty_loss = self.uncertainty_loss_weight * (sigma_weighted - sigma_target).abs().mean()

        return refined, uncertainty_loss
