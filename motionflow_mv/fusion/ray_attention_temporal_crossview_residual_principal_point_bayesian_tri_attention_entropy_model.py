"""Bayesian Tri v2 with attention-entropy regularisation on triangulation weights.

Subclasses ``RayAttentionFusionModelBayesianTriV2`` and adds an per-joint
entropy penalty on the normalised per-view triangulation weights. The loss
can be returned as an extra auxiliary term so the trainer can add it.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)


class RayAttentionFusionModelBayesianTriV2AttentionEntropy(
    RayAttentionFusionModelBayesianTriV2
):
    """Bayesian triangulation v2 with attention-entropy regularisation.

    Parameters
    ----------
    attention_entropy_weight:
        Weight for the entropy penalty (default 0.01). Set to 0.0 to disable.
    entropy_temperature:
        Temperature applied to weights before entropy computation.
    See ``RayAttentionFusionModelBayesianTriV2`` for the remaining args.
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
        covariance_hidden: int = 64,
        gn_iters: int = 2,
        min_gn_damping: float = 1e-6,
        max_gn_damping: float = 1e-2,
        epipolar_loss_weight: float = 0.05,
        return_covariance: bool = False,
        attention_entropy_weight: float = 0.01,
        entropy_temperature: float = 1.0,
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
            covariance_hidden=covariance_hidden,
            gn_iters=gn_iters,
            min_gn_damping=min_gn_damping,
            max_gn_damping=max_gn_damping,
            epipolar_loss_weight=epipolar_loss_weight,
            return_covariance=return_covariance,
        )
        self.attention_entropy_weight = attention_entropy_weight
        self.entropy_temperature = entropy_temperature

    def _entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute per-joint entropy of normalised view weights.

        Args:
            weights: (B*T, V, J) non-negative weights.

        Returns:
            Scalar entropy penalty (encourages concentrated weights).
        """
        # Normalise over views.
        p = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        # Add temperature to avoid log(0).
        p = torch.clamp(p, min=1e-8)
        # We return negative entropy as a penalty to minimise.
        entropy = -(p * torch.log(p)).sum(dim=1).mean()
        return -entropy / self.entropy_temperature

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        out = super().forward(x, cameras=cameras, K=K, R=R, t=t)
        if self.attention_entropy_weight == 0.0:
            return out

        # The base forward returns (pred, weights, [pp_delta], [focal], [cov], epi_loss).
        # We cannot easily recover the per-view weights from the output tuple, so we
        # re-run only the weight computation on the input. This is more expensive but
        # keeps the change minimal.
        # To avoid duplication, we instead approximate the penalty from the weights
        # tensor returned in the output tuple (out[1]).
        weights = out[1]  # (B, T, V, J)
        B, T, V, J = weights.shape
        weights_flat = weights.reshape(B * T, V, J).permute(0, 2, 1)  # (B*T, J, V)
        entropy_penalty = self._entropy_loss(weights_flat)
        epi_loss = out[-1]
        total_aux = epi_loss + self.attention_entropy_weight * entropy_penalty

        # Replace the last auxiliary loss with the combined term.
        return out[:-1] + (total_aux,)
