"""v55 Outlier-Robust Reliability (OR2).

An identity-at-init reliability module that sits between v45 adaptive geometry
fusion / v25 triangulation and v52 uncertainty-weighted triangulation.  It
predicts per-(view, joint) outlier scores from geometry and feature cues,
refines the incoming triangulation weights with a Cauchy M-estimator, and
passes the refined weights into v52 so that downstream physical/temporal
modules consume an already outlier-filtered estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutlierRobustReliabilityV55(nn.Module):
    """Outlier-robust reliability weighting for multi-view triangulation.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Number of camera views (shape hints only).
    hidden:
        Hidden dimension of the feature MLP.
    n_layers:
        Number of layers in the feature MLP.
    weight_type:
        One of ``per_view_joint``, ``per_view``, ``per_joint``.
    use_geometry_bias:
        Include reprojection/ray/epipolar geometry features.
    use_feature_bias:
        Include learned v45 feature-token statistics.
    identity_init:
        Zero-initialise the final MLP layer so the module is identity at init.
    min_weight:
        Floor on the refined weight.
    cauchy_gamma_init:
        Initial Cauchy scale ``γ``.
    residual_gate_init:
        Initial logit of the residual gate.
    use_entropy_reg:
        Add a small entropy regulariser to the auxiliary loss.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        weight_type: str = "per_view_joint",
        use_geometry_bias: bool = True,
        use_feature_bias: bool = True,
        identity_init: bool = True,
        min_weight: float = 0.05,
        cauchy_gamma_init: float = 1.0,
        residual_gate_init: float = -6.0,
        use_entropy_reg: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.weight_type = weight_type
        self.use_geometry_bias = use_geometry_bias
        self.use_feature_bias = use_feature_bias
        self.identity_init = identity_init
        self.min_weight = min_weight
        self.use_entropy_reg = use_entropy_reg

        if weight_type not in ("per_view_joint", "per_view", "per_joint"):
            raise ValueError(
                f"Unsupported v55 weight_type: {weight_type}. "
                "Choose from per_view_joint, per_view, per_joint."
            )

        feat_dim = 0
        if use_feature_bias:
            feat_dim += d * 3
        if use_geometry_bias:
            # reprojection residual, log residual, ray angle cosine,
            # epipolar distance, triangulation angle
            feat_dim += 5

        if feat_dim == 0:
            feat_dim = d
            self.use_feature_bias = True

        layers: list[nn.Module] = []
        for i in range(n_layers):
            is_last = i == n_layers - 1
            in_dim = feat_dim if i == 0 else hidden
            out_dim = 1 if is_last else hidden
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
        self.outlier_mlp = nn.Sequential(*layers)

        if identity_init:
            final_linear = self.outlier_mlp[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Cauchy scale (positive, learnable).
        self.cauchy_gamma = nn.Parameter(torch.tensor(cauchy_gamma_init, dtype=torch.float))

        # Gate that blends the original weight with the outlier-corrected weight.
        # At init, gate ~ 0 so the module is identity.
        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float))

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
        uv = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)
        uv = uv[..., :2] / uv[..., 2:3]
        residual = (uv - points_2d).norm(dim=-1)
        return residual

    def _compute_ray_angle(
        self,
        pred_3d: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine of angle between ray and (camera-center -> joint) vector.

        Returns (B, T, V, J).
        """
        B, T, J, _ = pred_3d.shape
        V = R.shape[2]
        # Camera centers.
        R_inv = R.transpose(-2, -1)
        c = -(R_inv @ t[..., None]).squeeze(-1)  # (B, T, V, 3)
        c = c.unsqueeze(3).expand(-1, -1, -1, J, -1)
        pred = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)
        to_joint = pred - c
        to_joint_norm = F.normalize(to_joint, dim=-1)
        # Principal axis in world frame.
        z_axis = R_inv[..., :, 2]  # (B, T, V, 3)
        z_axis = z_axis.unsqueeze(3).expand(-1, -1, -1, J, -1)
        cos_angle = (z_axis * to_joint_norm).sum(dim=-1)
        return cos_angle

    def _compute_geometry_features(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-(view, joint) geometry features (B, T, V, J, G)."""
        residual = self._compute_reprojection_residual(pred_3d, points_2d, K, R, t)
        log_residual = torch.log(residual.clamp(min=1e-6))
        ray_cos = self._compute_ray_angle(pred_3d, R, t)
        # Epipolar-ish distance: inverse depth residual proxy.
        B, T, J, _ = pred_3d.shape
        V = R.shape[2]
        pred = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)
        pred_cam = torch.matmul(R, pred.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        z = pred_cam[..., 2]
        inv_depth = 1.0 / z.clamp(min=1e-6)
        # Triangulation angle proxy: distance between the joint and the camera ray.
        R_inv = R.transpose(-2, -1)
        c = -(R_inv @ t[..., None]).squeeze(-1)  # (B, T, V, 3)
        c = c.unsqueeze(3).expand(-1, -1, -1, J, -1)
        to_joint = pred - c
        to_joint_norm = F.normalize(to_joint, dim=-1)
        z_axis = R_inv[..., :, 2].unsqueeze(3).expand(-1, -1, -1, J, -1)
        cross = torch.linalg.cross(to_joint_norm, z_axis, dim=-1)
        triang_dist = cross.norm(dim=-1)
        return torch.stack([residual, log_residual, ray_cos, inv_depth, triang_dist], dim=-1)

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

        if self.use_geometry_bias:
            geom = self._compute_geometry_features(pred_3d_init, points_2d, K, R, t)
            feats.append(geom)

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
        weights_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict outlier-robust reliability weights.

        Args:
            features: (B, T, V, J, d) feature tokens.
            points_2d: (B, T, V, J, 2) 2-D keypoints.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            pred_3d_init: (B, T, J, 3) initial 3-D estimate.
            view_mask: optional (B, T, V) bool mask. True = view valid.
            weights_init: optional (B, T, V, J) initial weights. Defaults to 1.

        Returns:
            weights_orr: (B, T, V, J) refined reliability weights.
            orr_loss: scalar auxiliary loss.
        """
        B, T, V, J, d = features.shape
        feat = self._build_features(features, points_2d, pred_3d_init, K, R, t)

        if self.weight_type == "per_view_joint":
            logit = self.outlier_mlp(feat).squeeze(-1)  # (B, T, V, J)
        elif self.weight_type == "per_view":
            feat_pooled = feat.mean(dim=3)  # (B, T, V, feat_dim)
            logit = self.outlier_mlp(feat_pooled).squeeze(-1)  # (B, T, V)
            logit = logit.unsqueeze(-1).expand(-1, -1, -1, J)
        else:  # per_joint
            feat_pooled = feat.mean(dim=2)  # (B, T, J, feat_dim)
            logit = self.outlier_mlp(feat_pooled).squeeze(-1)  # (B, T, J)
            logit = logit.unsqueeze(2).expand(-1, -1, V, -1)

        # Identity at init: logit ~ 0 -> outlier_score ~ 0.5.
        outlier_score = torch.sigmoid(logit)

        # Cauchy M-estimator inlier likelihood.
        gamma = self.cauchy_gamma.abs().clamp(min=1e-6)
        residual_cauchy = outlier_score / gamma
        inlier_likelihood = 1.0 / (1.0 + residual_cauchy * residual_cauchy)

        if weights_init is None:
            weights_prior = torch.ones(B, T, V, J, device=features.device, dtype=features.dtype)
        else:
            weights_prior = weights_init

        # Blend: at init the gate is ~0, so weights_orr ~ weights_prior.
        gate = torch.sigmoid(self.residual_gate)
        weights_orr = (1.0 - gate) * weights_prior + gate * (weights_prior * inlier_likelihood)
        weights_orr = weights_orr.clamp(min=self.min_weight, max=1.0)

        if view_mask is not None:
            weights_orr = weights_orr * view_mask.unsqueeze(-1).float()

        # Auxiliary loss: encourage inlier_likelihood to follow reprojection residual.
        with torch.no_grad():
            reproj = self._compute_reprojection_residual(pred_3d_init, points_2d, K, R, t)
            target = torch.exp(-reproj / 5.0).clamp(min=self.min_weight, max=1.0)
        consistency_loss = F.mse_loss(weights_orr, target)

        orr_loss = consistency_loss
        if self.use_entropy_reg:
            weights_norm = weights_orr / (weights_orr.sum(dim=2, keepdim=True).clamp(min=1e-6))
            entropy = -(weights_norm * torch.log(weights_norm + 1e-6)).sum(dim=2).mean()
            orr_loss = orr_loss - 0.01 * entropy

        return weights_orr, orr_loss
