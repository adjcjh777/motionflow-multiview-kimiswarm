"""v31: per-joint gated skeleton-graph residual refiner.

Builds on the v30 skeleton-graph refiner but adds a learned per-joint gate so
that distal joints can receive larger corrections while the torso stays
anchored. The gate is initialised to a small positive value for stable
training and is modulated by the magnitude of the input residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .skeleton_graph_residual_refiner import SkeletonGraphResidualRefiner


class SkeletonGraphResidualRefinerV31(nn.Module):
    """Skeleton-graph residual refiner with per-joint gated residuals.

    Parameters
    ----------
    j:
        Number of joints.
    in_dim:
        Input feature dimension.
    hidden_dim:
        Hidden dimension of the graph layers.
    num_layers:
        Number of graph message-passing layers.
    gate_init:
        Initial gate value for each joint.
    """

    def __init__(
        self,
        j: int,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.base = SkeletonGraphResidualRefiner(
            j=j,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        # Per-joint gate, initialised small for identity-like behaviour.
        self.gate = nn.Parameter(torch.full((j, 1), gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args
        ----
        x:
            ``(B, J, in_dim)`` concatenation of pooled feature and raw 3-D pose.

        Returns
        -------
        ``(B, J, 3)`` gated residual correction.
        """
        residual = self.base(x)  # (B, J, 3)
        # Modulate by per-joint gate (soft, positive) and input magnitude.
        magnitude = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, J, 1)
        gate = torch.sigmoid(self.gate)  # (J, 1)
        # Gate scale is joint-specific and softly normalised by input magnitude.
        return residual * gate * torch.tanh(magnitude)
