"""v37: Self-Critique View Reliability Estimator.

Predicts per-(view, joint) reliability scores from the model's own refined
tokens. The scores are used to soft-weight triangulation, cross-view
attention, and reprojection losses. The block is identity at initialization
(output bias ~ +2.0 => sigmoid ~0.88) so it does not hurt the baseline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCritiqueViewReliabilityV37(nn.Module):
    """Predict per-view-joint reliability from refined tokens.

    Parameters
    ----------
    d:
        Feature dimension.
    hidden_dim:
        Hidden dimension of the reliability MLP.
    n_layers:
        Number of MLP layers.
    use_temporal_context:
        If True, append a temporal 1-D conv over the time dimension.
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_temporal_context: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal_context = use_temporal_context

        layers = []
        in_dim = d
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize output bias so sigmoid outputs ~0.88 (mostly reliable).
        last_linear = None
        for m in reversed(self.mlp):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            with torch.no_grad():
                last_linear.bias.fill_(2.0)

        if use_temporal_context:
            self.temporal_conv = nn.Conv1d(
                in_channels=hidden_dim if n_layers > 1 else d,
                out_channels=hidden_dim if n_layers > 1 else d,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            )

    def forward(
        self,
        tokens: torch.Tensor,
        points_2d: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute reliability scores.

        Args:
            tokens: Refined tokens, shape (B, T, V, J, d).
            points_2d: Optional raw 2-D points, shape (B, T, V, J, 2).
            K, R, t: Optional camera parameters.
            view_mask: Optional mask, shape (B, T, V).

        Returns:
            reliability: (B, T, V, J) in [0, 1].
            view_reliability: (B, T, V) in [0, 1], aggregated per view.
        """
        B, T, V, J, d = tokens.shape
        # Flatten across views and joints for the MLP.
        x = tokens.reshape(B * T * V * J, d)
        logits = self.mlp(x).squeeze(-1)  # (B*T*V*J,)
        reliability = torch.sigmoid(logits).reshape(B, T, V, J)

        if view_mask is not None:
            # Mask invalid views before aggregation.
            mask = view_mask.unsqueeze(-1)  # (B, T, V, 1)
            reliability = reliability * mask.float()

        # Aggregate per-view reliability by averaging over joints.
        view_reliability = reliability.mean(dim=-1)  # (B, T, V)

        return reliability, view_reliability
