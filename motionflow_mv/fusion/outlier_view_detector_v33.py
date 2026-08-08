"""v33 learned outlier-view detector with adaptive per-joint/part/domain thresholds.

The detector refines the existing v25 ``OutlierViewDetector`` by adding:

* a feature-aware residual adjustment,
* learnable per-joint, per-body-part and per-domain threshold/softness scales,
* an optional supervised BCE loss on the augmentation outlier mask.

All gates are initialised so that the block is identity at init: the residual
adjustment is gated to zero, the final soft-down-weight gate is zero, and all
adaptive scales start at one.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.outlier_view_detector import compute_reprojection_residual


def _generic_part_ids(j: int, num_parts: int = 5, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a deterministic part id for each joint.

    The mapping is generic rather than dataset-specific: it spreads joints
    across ``num_parts`` body regions so that learned per-part scales can
    specialise.  A more semantic mapping can be injected by the caller if
    desired.
    """
    ids = torch.arange(j, device=device) % num_parts
    return ids


class OutlierViewDetectorV33(nn.Module):
    """Learned, feature-aware outlier-view detector.

    Parameters
    ----------
    z_thresh:
        Base robust z-score threshold.
    soft_beta:
        Base softness of the exponential down-weighting.
    min_mad:
        Floor on the MAD-derived standard deviation.
    num_joints:
        Number of joints, used for per-joint scale parameters.
    num_parts:
        Number of body-part groups for per-part scales.
    num_domains:
        Number of dataset domains for per-domain scales.
    feature_dim:
        Dimension of the per-joint feature tokens.
    feature_hidden:
        Hidden dimension of the feature-aware residual MLP.
    use_feature_gate:
        If True, use per-view feature tokens to adjust the residual.
    use_part_scale:
        If True, learn per-part threshold/softness scales.
    use_domain_scale:
        If True, learn per-domain threshold/softness scales.
    supervised_weight:
        Weight for the optional BCE loss on augmentation masks.
    """

    def __init__(
        self,
        z_thresh: float = 3.0,
        soft_beta: float = 1.0,
        min_mad: float = 0.5,
        num_joints: int = 17,
        num_parts: int = 5,
        num_domains: int = 3,
        feature_dim: int = 64,
        feature_hidden: int = 64,
        use_feature_gate: bool = True,
        use_part_scale: bool = True,
        use_domain_scale: bool = True,
        supervised_weight: float = 0.1,
    ):
        super().__init__()
        self.base_z_thresh = z_thresh
        self.base_soft_beta = soft_beta
        self.min_mad = min_mad
        self.use_feature_gate = use_feature_gate
        self.use_part_scale = use_part_scale
        self.use_domain_scale = use_domain_scale
        self.supervised_weight = supervised_weight

        # Final down-weight gate.  Initialised to -6 -> sigmoid ~0 so the block is identity at init.
        self.residual_scale = nn.Parameter(torch.tensor(-6.0))

        # Per-joint threshold/softness scales.
        self.z_scale_joint = nn.Parameter(torch.ones(num_joints))
        self.beta_scale_joint = nn.Parameter(torch.ones(num_joints))

        # Per-part threshold/softness scales.
        if self.use_part_scale:
            self.z_scale_part = nn.Embedding(num_parts, 1)
            self.beta_scale_part = nn.Embedding(num_parts, 1)
            nn.init.ones_(self.z_scale_part.weight)
            nn.init.ones_(self.beta_scale_part.weight)
        else:
            self.z_scale_part = None
            self.beta_scale_part = None

        # Per-domain threshold/softness scales.
        if self.use_domain_scale:
            self.z_scale_domain = nn.Embedding(num_domains, 1)
            self.beta_scale_domain = nn.Embedding(num_domains, 1)
            nn.init.ones_(self.z_scale_domain.weight)
            nn.init.ones_(self.beta_scale_domain.weight)
        else:
            self.z_scale_domain = None
            self.beta_scale_domain = None

        # Feature-aware residual adjustment.
        # Operates directly on per-joint feature tokens (..., d) and produces a
        # per-joint residual offset (delta) and a per-joint gate (alpha).
        if self.use_feature_gate:
            self.feature_dim = feature_dim
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_hidden),
                nn.ReLU(),
                nn.Linear(feature_hidden, 2),
            )
            # Identity at init: delta=0, alpha=-3 -> sigmoid(alpha) ~ 0.
            nn.init.zeros_(self.feature_mlp[-1].bias)
            self.feature_mlp[-1].bias.data[1] = -3.0
        else:
            self.feature_dim = feature_dim
            self.feature_mlp = None

    def _compute_part_ids(self, j: int, device: torch.device) -> torch.Tensor:
        return _generic_part_ids(j, num_parts=self.z_scale_part.num_embeddings if self.z_scale_part is not None else 5, device=device)

    def _adaptive_scales(
        self,
        domain_ids: Optional[torch.Tensor],
        num_joints: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute effective z_thresh and soft_beta scales for each joint.

        Returns
        -------
        z_scale, beta_scale: (1, 1, 1, J) tensors.
        """
        j_ids = torch.arange(num_joints, device=device)
        z_scale = self.z_scale_joint[j_ids]  # (J,)
        beta_scale = self.beta_scale_joint[j_ids]

        if self.use_part_scale and self.z_scale_part is not None and self.beta_scale_part is not None:
            part_ids = self._compute_part_ids(num_joints, device=device)
            z_scale = z_scale * self.z_scale_part(part_ids).squeeze(-1)
            beta_scale = beta_scale * self.beta_scale_part(part_ids).squeeze(-1)

        if self.use_domain_scale and domain_ids is not None and self.z_scale_domain is not None and self.beta_scale_domain is not None:
            # domain_ids: (B,) or (B, 1); average scale over the batch for broadcasting.
            if domain_ids.dim() == 1:
                domain_ids = domain_ids[:, None]
            # Use the first id per sample, shape (B, 1)
            dids = domain_ids[:, 0].long()
            z_domain = self.z_scale_domain(dids).squeeze(-1)  # (B,)
            beta_domain = self.beta_scale_domain(dids).squeeze(-1)
            z_scale = z_scale * z_domain.mean()
            beta_scale = beta_scale * beta_domain.mean()

        return z_scale.view(1, 1, 1, -1), beta_scale.view(1, 1, 1, -1)

    def _feature_adjusted_residual(
        self,
        residual: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply feature-aware residual adjustment.

        Args:
            residual: (B, T, V, J)
            features: (B, T, V, J, d)

        Returns:
            adjusted: (B, T, V, J)
        """
        out = self.feature_mlp(features)  # (B, T, V, J, 2)
        delta = out[..., 0]
        alpha = out[..., 1]
        return residual + torch.sigmoid(alpha) * delta

    def forward(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        outlier_label: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft outlier weights and auxiliary loss.

        Args:
            pred_3d: (B, T, J, 3)
            points_2d: (B, T, V, J, 2)
            K: (B, T, V, 3, 3)
            R: (B, T, V, 3, 3)
            t: (B, T, V, 3)
            features: optional (B, T, V, J, d)
            domain_ids: optional (B,) or (B, T)
            view_mask: optional (B, T, V)
            outlier_label: optional (B, T, V, J) target mask (1=outlier)

        Returns:
            weights: (B, T, V, J)
            aux_loss: scalar
        """
        residual = compute_reprojection_residual(pred_3d, points_2d, K, R, t)  # (B, T, V, J)

        if self.use_feature_gate and features is not None:
            residual = self._feature_adjusted_residual(residual, features)

        # Robust statistics over views.
        median = residual.median(dim=2, keepdim=True)[0]  # (B, T, 1, J)
        mad = (residual - median).abs().median(dim=2, keepdim=True)[0]
        mad_std = 1.4826 * mad
        z_score = (residual - median) / (mad_std + self.min_mad + 1e-6)  # (B, T, V, J)

        # Effective thresholds.
        num_joints = residual.shape[-1]
        device = residual.device
        z_scale, beta_scale = self._adaptive_scales(domain_ids, num_joints, device)
        z_thresh_eff = self.base_z_thresh * z_scale
        soft_beta_eff = self.base_soft_beta * beta_scale

        # Soft down-weighting.
        margin = torch.clamp(z_score - z_thresh_eff, min=0.0)
        consensus_weights = torch.exp(-soft_beta_eff * margin)

        gate = torch.sigmoid(self.residual_scale)
        weights = 1.0 - gate * (1.0 - consensus_weights)
        weights = weights.clamp(min=0.0, max=1.0)

        if view_mask is not None:
            weights = weights * view_mask[..., None]

        # Optional supervised loss.
        aux_loss = torch.tensor(0.0, device=device, dtype=residual.dtype)
        if outlier_label is not None and self.supervised_weight > 0:
            # Target: 1 - weight should be close to outlier_label.
            pred_outlier_prob = 1.0 - weights
            bce = F.binary_cross_entropy(
                pred_outlier_prob,
                outlier_label.float(),
                reduction="none",
            )
            if view_mask is not None:
                bce = bce * view_mask[..., None]
            aux_loss = self.supervised_weight * bce.mean()

        return weights, aux_loss
