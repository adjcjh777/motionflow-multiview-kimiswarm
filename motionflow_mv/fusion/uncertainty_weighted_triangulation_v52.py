"""v52 Uncertainty-Weighted Triangulation (UWT).

A learnable triangulation module that predicts per-view (and optionally per-joint)
precision weights, re-triangulates with a weighted DLT, and applies a small
gated residual correction.  The module is identity at initialization: the
precision MLP and residual MLP final layers are zero-initialized, and the
residual gate is initialised to 0, so ``pred_3d == pred_3d_init`` until the
module is trained.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.utils.geometry import weighted_dlt_triangulate


class UncertaintyWeightedTriangulationV52(nn.Module):
    """Learnable uncertainty-weighted triangulation.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Number of camera views (for shape hints only).
    hidden:
        Hidden dimension of the precision MLP.
    n_layers:
        Number of layers in the precision MLP.
    weight_type:
        One of ``per_view_joint``, ``per_view``, ``per_joint``.
    temperature:
        Temperature applied to log-precision before the sigmoid.
    use_geometry_bias:
        Include reprojection-based geometry features.
    use_feature_bias:
        Include feature-token mean/std statistics.
    identity_init:
        Zero-initialize the final precision MLP layer so weights start at ~0.5.
    min_weight:
        Floor on the predicted triangulation weight.
    residual_gate_init:
        Initial value of the residual correction gate (0.0 = identity at init).
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        weight_type: str = "per_view_joint",
        temperature: float = 1.0,
        use_geometry_bias: bool = True,
        use_feature_bias: bool = True,
        identity_init: bool = True,
        min_weight: float = 0.05,
        residual_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.weight_type = weight_type
        self.temperature = temperature
        self.use_geometry_bias = use_geometry_bias
        self.use_feature_bias = use_feature_bias
        self.identity_init = identity_init
        self.min_weight = min_weight

        if weight_type not in ("per_view_joint", "per_view", "per_joint"):
            raise ValueError(
                f"Unsupported v52 weight_type: {weight_type}. "
                "Choose from per_view_joint, per_view, per_joint."
            )

        # Build feature dimension.
        feat_dim = 0
        if use_feature_bias:
            # raw token + mean over views + std over views
            feat_dim += d * 3
        if use_geometry_bias:
            # reprojection residual + log residual
            feat_dim += 2

        if feat_dim == 0:
            # Fallback: at least use the raw token so the module has input.
            feat_dim = d
            self.use_feature_bias = True

        # Precision MLP.
        layers: list[nn.Module] = []
        for i in range(n_layers):
            is_last = i == n_layers - 1
            in_dim = feat_dim if i == 0 else hidden
            out_dim = 1 if is_last else hidden
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
        self.precision_mlp = nn.Sequential(*layers)

        if identity_init:
            # Zero-init final layer -> log_precision ~ 0 -> sigmoid(0) = 0.5.
            final_linear = self.precision_mlp[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Small gated residual correction around the input estimate.
        self.residual_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        # Zero-initialise the final residual projection for identity at init.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init))

    def _compute_reprojection_residual(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-view per-joint reprojection residual norm (B, T, V, J)."""
        B, T, V, J, _ = points_2d.shape
        X = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B, T, V, J, 3)
        X = X.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
        X_cam = torch.matmul(R, X) + t[..., None]  # (B, T, V, 3, J)
        X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
        Z = X_cam[..., 2:3].clamp(min=1e-6)
        X_norm = X_cam / Z
        uv = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)  # (B, T, V, J, 3)
        uv = uv[..., :2] / uv[..., 2:3]
        residual = (uv - points_2d).norm(dim=-1)  # (B, T, V, J)
        return residual

    def _build_features(
        self,
        features: torch.Tensor,
        points_2d: torch.Tensor,
        pred_3d_init: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-(view, joint) feature vectors (B, T, V, J, feat_dim)."""
        B, T, V, J, d = features.shape
        feats: list[torch.Tensor] = []

        if self.use_feature_bias:
            mean_view = features.mean(dim=2, keepdim=True).expand(-1, -1, V, -1, -1)
            std_view = features.std(dim=2, keepdim=True, unbiased=False).expand(-1, -1, V, -1, -1)
            feats.append(features)
            feats.append(mean_view)
            feats.append(std_view)
        else:
            feats.append(features)

        if self.use_geometry_bias:
            residual = self._compute_reprojection_residual(pred_3d_init, points_2d, K, R, t)
            log_residual = torch.log(residual.clamp(min=1e-6))
            feats.append(residual.unsqueeze(-1))
            feats.append(log_residual.unsqueeze(-1))

        return torch.cat(feats, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        pred_3d_init: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        weights_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict precision weights, re-triangulate, and refine.

        Args:
            features: (B, T, V, J, d) feature tokens.
            points_2d: (B, T, V, J, 2) 2-D keypoints.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            pred_3d_init: (B, T, J, 3) initial 3-D estimate.
            view_mask: optional (B, T, V) bool mask. True = view valid.
            domain_id: optional (B,) integer domain labels (unused but kept
                for API compatibility).
            weights_prior: optional (B, T, V, J) prior weights from an upstream
                reliability module (e.g. v55 OR2).

        Returns:
            pred_3d: (B, T, J, 3) refined 3-D estimate.
            uwt_loss: scalar auxiliary loss.
            weights: (B, T, V, J) predicted triangulation weights.
            log_precision: (B, T, V, J) or broadcasted log-precision values.
        """
        B, T, V, J, d = features.shape
        feat = self._build_features(features, points_2d, pred_3d_init, K, R, t)

        # Predict log-precision according to weight_type.
        if self.weight_type == "per_view_joint":
            log_precision = self.precision_mlp(feat).squeeze(-1)  # (B, T, V, J)
        elif self.weight_type == "per_view":
            # Pool over joints and predict one logit per view.
            feat_pooled = feat.mean(dim=3)  # (B, T, V, feat_dim)
            log_precision = self.precision_mlp(feat_pooled).squeeze(-1)  # (B, T, V)
            log_precision = log_precision.unsqueeze(-1).expand(-1, -1, -1, J)
        else:  # per_joint
            feat_pooled = feat.mean(dim=2)  # (B, T, J, feat_dim)
            log_precision = self.precision_mlp(feat_pooled).squeeze(-1)  # (B, T, J)
            log_precision = log_precision.unsqueeze(2).expand(-1, -1, V, -1)

        # Stable precision weights.
        weights = torch.sigmoid(log_precision / self.temperature)
        weights = weights.clamp(min=self.min_weight, max=1.0)

        # Apply optional upstream reliability prior (e.g. v55 OR2).
        if weights_prior is not None:
            weights = weights * weights_prior
            weights = weights.clamp(min=self.min_weight, max=1.0)

        # Apply view mask to zero out missing views.
        if view_mask is not None:
            weights = weights * view_mask.unsqueeze(-1).float()

        # Weighted DLT triangulation.
        pred_3d_ref = weighted_dlt_triangulate(
            points_2d,
            K,
            R,
            t,
            weights=weights,
            view_mask=view_mask,
            eps=1e-6,
            damping=1e-4,
        )

        # Gated residual correction around the initial estimate.
        residual = self.residual_mlp(pred_3d_ref - pred_3d_init)
        pred_3d = pred_3d_init + self.residual_gate * residual

        # Auxiliary loss: consistency with reprojection residual + entropy.
        with torch.no_grad():
            residual = self._compute_reprojection_residual(pred_3d_init, points_2d, K, R, t)
            target = torch.exp(-residual / 5.0).clamp(min=self.min_weight, max=1.0)

        consistency_loss = F.mse_loss(weights, target)

        # Entropy regularisation: maximise entropy of weights per joint.
        # Normalised to be scale-invariant.
        weights_norm = weights / (weights.sum(dim=2, keepdim=True).clamp(min=1e-6))
        entropy = -(weights_norm * torch.log(weights_norm + 1e-6)).sum(dim=2).mean()
        entropy_loss = -entropy  # maximise entropy -> minimise negative entropy

        uwt_loss = consistency_loss + 0.01 * entropy_loss

        return pred_3d, uwt_loss, weights.detach(), log_precision.detach()
