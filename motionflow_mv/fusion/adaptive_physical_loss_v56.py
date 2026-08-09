"""v56 Adaptive Physical Loss Weighting (APL) for v53 PSC.

Learns a per-sample scalar weight for the v53 PSC loss from triangulation
uncertainty and the current pose. Identity-at-init: the predicted weight starts
near 1.0 so a v53 checkpoint loaded with v56 enabled does not change the initial
PSC loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePhysicalLossV56(nn.Module):
    """Predict a per-sample scalar weight for the v53 PSC loss.

    Parameters
    ----------
    hidden:
        Hidden dimension of the weighting MLP.
    identity_init:
        If True, initialise so that the output weight is 1.0 at the start.
    """

    def __init__(
        self,
        hidden: int = 32,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden

        # Input: uncertainty mean/std + pose std.
        in_dim = 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        if identity_init:
            final = self.mlp[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        uncertainty: torch.Tensor,
        pred_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample positive weight for the PSC loss.

        Args:
            uncertainty: (B, T, V, J) per-view-joint uncertainty (0 = certain).
            pred_3d: (B, T, J, 3) current pose.

        Returns:
            weight: (B,) positive scalar around 1.0 at init.
        """
        u_mean = uncertainty.mean(dim=(1, 2, 3))  # (B,)
        u_std = uncertainty.std(dim=(1, 2, 3), unbiased=False)  # (B,)
        p_std = pred_3d.std(dim=(1, 2, 3), unbiased=False)  # (B,)

        feat = torch.stack([u_mean, u_std, p_std], dim=1)  # (B, 3)
        logit = self.mlp(feat).squeeze(-1)  # (B,)
        weight = F.softplus(logit) + 1.0
        return weight
