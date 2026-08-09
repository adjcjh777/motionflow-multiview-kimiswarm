"""v59 View-Count-Conditioned Sparse-View Reliability (VCC-SVR).

A small auxiliary head that predicts an additive offset to v52's log-precision
using the number of visible views.  It is identity at initialization: the final
MLP layer is zero-initialized, so the offset is ~0 until the module is trained.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ViewCountConditionedReliabilityV59(nn.Module):
    """Predict a per-(view, joint) log-precision offset from the view count.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Maximum number of camera views (for shape hints only).
    hidden:
        Hidden dimension of the view-count MLP.
    n_layers:
        Number of layers in the view-count MLP.
    max_views:
        Maximum supported view count for the learned count embedding.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        hidden: int = 32,
        n_layers: int = 2,
        max_views: int = 8,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.hidden = hidden
        self.max_views = max_views

        # Learnable embedding for each possible view-count value.
        self.view_count_embed = nn.Embedding(max_views + 1, hidden)

        # Per-view MLP: pooled joint statistics + count embedding -> offset scalar.
        input_dim = d * 2 + hidden
        layers: list[nn.Module] = []
        for i in range(n_layers):
            is_last = i == n_layers - 1
            in_dim = input_dim if i == 0 else hidden
            out_dim = 1 if is_last else hidden
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        # Zero-initialize the final layer so the offset is ~0 at initialization.
        final_linear = self.mlp[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        features: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``log_precision_offset`` of shape (B, T, V, J).

        Args:
            features: (B, T, V, J, d) feature tokens.
            view_mask: optional (B, T, V) bool mask. True = view visible.

        Returns:
            offset: (B, T, V, J) additive log-precision offset.
        """
        B, T, V, J, d = features.shape

        if view_mask is None:
            view_mask = torch.ones(B, T, V, dtype=torch.bool, device=features.device)
        else:
            view_mask = view_mask.bool()

        # Number of visible views per (B, T), clamped to the embedding table size.
        num_visible = view_mask.sum(dim=-1).long()  # (B, T)
        num_visible = num_visible.clamp(0, self.max_views)
        count_embed = self.view_count_embed(num_visible)  # (B, T, hidden)

        # Per-view joint statistics: mean and std over joints.
        mean_joint = features.mean(dim=3)  # (B, T, V, d)
        std_joint = features.std(dim=3, unbiased=False)  # (B, T, V, d)
        per_view_feat = torch.cat([mean_joint, std_joint], dim=-1)  # (B, T, V, 2d)

        # Broadcast the count embedding to each view and concatenate.
        count_embed = count_embed.unsqueeze(2).expand(-1, -1, V, -1)  # (B, T, V, hidden)
        mlp_input = torch.cat([per_view_feat, count_embed], dim=-1)  # (B, T, V, 2d+hidden)

        # Predict a single scalar per view and expand to all joints.
        offset = self.mlp(mlp_input)  # (B, T, V, 1)
        offset = offset.expand(-1, -1, -1, J)  # (B, T, V, J)
        return offset
