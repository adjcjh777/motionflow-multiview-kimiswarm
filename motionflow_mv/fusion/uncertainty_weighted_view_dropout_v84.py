"""v84: Uncertainty-Weighted View Dropout (UWVD).

A lightweight per-view uncertainty head that predicts log-variance from geometry
cues (reprojection error + epipolar residual) and pooled ray tokens.  The
uncertainty is turned into weights via softmax and used to:

1. Stochastically drop views during training (more uncertain views are dropped
   more often).
2. Reweight views during triangulation.

The whole block is gated by a zero-initialised residual so v25 behaviour is
preserved at the start of training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedViewDropoutV84(nn.Module):
    """Predict per-view uncertainty and produce triangulation/dropout weights.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Maximum number of camera views.
    n_joints:
        Number of joints.
    hidden:
        Hidden dimension of the uncertainty MLP.
    weight_type:
        ``per_view`` pools across joints and predicts one weight per view;
        ``per_view_joint`` keeps per-joint predictions.
    use_reprojection:
        Feed reprojection error residual into the uncertainty MLP.
    use_epipolar:
        Feed epipolar-style residual into the uncertainty MLP.
    dropout_prob:
        Base view-dropout probability during training.
    min_weight:
        Minimum triangulation weight after softmax clamping.
    identity_init:
        Zero-initialise the final MLP layer so initial weights are uniform.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 4,
        n_joints: Optional[int] = None,
        hidden: int = 64,
        weight_type: str = "per_view",
        use_reprojection: bool = True,
        use_epipolar: bool = True,
        dropout_prob: float = 0.1,
        min_weight: float = 0.05,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        if weight_type not in ("per_view", "per_view_joint"):
            raise ValueError(
                f"weight_type must be 'per_view' or 'per_view_joint', got {weight_type}"
            )

        self.d = d
        self.n_views = n_views
        self.n_joints = n_joints
        self.weight_type = weight_type
        self.use_reprojection = use_reprojection
        self.use_epipolar = use_epipolar
        self.dropout_prob = dropout_prob
        self.min_weight = min_weight

        in_dim = d + (1 if use_reprojection else 0) + (1 if use_epipolar else 0)
        if in_dim == d:
            raise ValueError("At least one of use_reprojection/use_epipolar must be True")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

        if identity_init:
            final_linear = self.mlp[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Global residual gate; sigmoid(-6.0) ~ 0.002, near-identity at init.
        self.residual_gate = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32))

    def _project(
        self,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Project world 3D points into each view. Returns (B, T, V, J, 2)."""
        X = pred_3d.unsqueeze(2).unsqueeze(-1)  # (B, T, 1, J, 3, 1)
        R = R.unsqueeze(3)  # (B, T, V, 1, 3, 3)
        t = t.unsqueeze(-2)  # (B, T, V, 1, 3)
        X_cam = (R @ X).squeeze(-1) + t  # (B, T, V, J, 3)
        z = X_cam[..., 2:3].clamp(min=1e-6)
        proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
        uv = proj[..., :2] / z
        return uv

    def _reprojection_error(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-view reprojection error. Returns (B, T, V, J)."""
        uv = self._project(pred_3d, K, R, t)
        residual = (uv - points_2d).norm(dim=-1)
        return residual

    def _epipolar_residual(self, reproj_error: torch.Tensor) -> torch.Tensor:
        """Per-view deviation from mean reprojection error. Returns (B, T, V, J)."""
        mean_error = reproj_error.mean(dim=2, keepdim=True)  # (B, T, 1, J)
        return reproj_error - mean_error

    def _compute_view_weights(
        self,
        tokens: torch.Tensor,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return triangulation weights (B, T, V) or (B, T, V, J).

        Lower uncertainty -> higher weight. Invalid views are masked to zero.
        """
        B, T, V, J, d = tokens.shape
        # Pool tokens per view: (B, T, V, d)
        pooled = tokens.mean(dim=3)  # (B, T, V, d)

        features = [pooled]
        if self.use_reprojection or self.use_epipolar:
            reproj = self._reprojection_error(pred_3d, points_2d, K, R, t)  # (B, T, V, J)
            if self.use_reprojection:
                # Pool reprojection error across joints: (B, T, V, 1)
                features.append(reproj.mean(dim=-1, keepdim=True))
            if self.use_epipolar:
                epipolar = self._epipolar_residual(reproj)  # (B, T, V, J)
                features.append(epipolar.mean(dim=-1, keepdim=True))

        x = torch.cat(features, dim=-1)  # (B, T, V, in_dim)

        if self.weight_type == "per_view":
            # Predict a single log-variance per view.
            log_var = self.mlp(x).squeeze(-1)  # (B, T, V)
        else:
            # Predict per-view-joint log-variance.
            # Broadcast pooled features to per-joint.
            x_j = x.unsqueeze(3).expand(-1, -1, -1, J, -1)  # (B, T, V, J, in_dim)
            log_var = self.mlp(x_j).squeeze(-1)  # (B, T, V, J)

        # Convert log-variance to weights: lower variance -> higher weight.
        weights = F.softmax(-log_var, dim=2)  # softmax over views

        if view_mask is not None:
            if weights.dim() == 3:
                vm = view_mask.float()  # (B, T, V)
            else:
                vm = view_mask.unsqueeze(3)  # (B, T, V, 1)
            weights = weights * vm
            weights = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-6))

        weights = weights.clamp(min=self.min_weight)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        return weights

    def _stochastic_dropout(
        self,
        weights: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply training-time uncertainty-weighted view dropout.

        More uncertain views (lower weights) are dropped with higher probability.
        Returns a dropout mask to multiply with view_mask / triangulation weights.
        """
        if not self.training or self.dropout_prob <= 0.0:
            return weights

        # Normalise weights to get per-view probability of being kept.
        normalized = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        # Drop probability is higher for low-weight views.
        drop_prob = self.dropout_prob * (1.0 - normalized)  # (B, T, V) or (B, T, V, J)
        keep_mask = torch.bernoulli(1.0 - drop_prob).to(weights.device)

        if view_mask is not None:
            keep_mask = keep_mask * view_mask.float()

        dropped_weights = weights * keep_mask
        # Renormalise across kept views.
        dropped_weights = dropped_weights / dropped_weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        return dropped_weights

    def forward(
        self,
        tokens: torch.Tensor,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply uncertainty-weighted view dropout.

        Args:
            tokens: (B, T, V, J, d)
            pred_3d: (B, T, J, 3)
            points_2d: (B, T, V, J, 2)
            K: (B, T, V, 3, 3)
            R: (B, T, V, 3, 3)
            t: (B, T, V, 3)
            view_mask: (B, T, V)

        Returns:
            refined_tokens: (B, T, V, J, d)
            view_weights: (B, T, V) or (B, T, V, J)
        """
        view_weights = self._compute_view_weights(
            tokens, pred_3d, points_2d, K, R, t, view_mask=view_mask
        )

        # Training-time stochastic dropout.
        tri_weights = self._stochastic_dropout(view_weights, view_mask=view_mask)

        # Residual update on tokens (near-identity at init).
        gate = torch.sigmoid(self.residual_gate)
        # Broadcast tri_weights to token dimension.
        weight_signal = tri_weights
        if weight_signal.dim() == 3:
            weight_signal = weight_signal.unsqueeze(3).unsqueeze(-1)  # (B, T, V, 1, 1)
        else:
            weight_signal = weight_signal.unsqueeze(-1)  # (B, T, V, J, 1)
        refined_tokens = tokens + gate * weight_signal * tokens

        return refined_tokens, view_weights
