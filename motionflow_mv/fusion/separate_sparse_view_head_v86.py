"""v86: Separate sparse-view head for variable-view triangulation.

A lightweight, dedicated head that is applied whenever the number of active
views is less than the maximum (``k < n_views``).  It pools the per-view ray
tokens over the active views, injects a view-count embedding, and predicts a
small residual correction around the initial DLT estimate.

The head is identity-at-init: the final MLP layer is zero-initialised and the
residual gate starts at zero, so the initial 3-D estimate passes through
unchanged.  Because it is only used for sparse samples, the full-view path is
not affected and the main geometry-fusion parameters do not receive gradients
from sparse-view examples.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparateSparseViewHeadV86(nn.Module):
    """Dedicated head for sparse (``k < n_views``) multi-view inputs.

    Parameters
    ----------
    d:
        Ray-token dimension (must match the v25 ray-token dimension).
    n_views:
        Maximum number of camera views (used to size the count embedding).
    n_joints:
        Number of body joints.
    hidden:
        Hidden dimension of the correction MLP.
    n_layers:
        Number of MLP layers (excluding the final output layer).
    dropout:
        Dropout probability inside the MLP.
    use_count_embedding:
        If True, add a learned embedding keyed by the active view count.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 4,
        n_joints: int = 17,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_count_embedding: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.d = d
        self.n_views = n_views
        self.n_joints = n_joints
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_count_embedding = use_count_embedding

        if use_count_embedding:
            self.count_embed = nn.Embedding(n_views + 1, d)
            nn.init.zeros_(self.count_embed.weight)
        else:
            self.count_embed = None

        # Aggregate active-view tokens to per-joint features, then correct the
        # initial 3-D estimate with a small residual MLP.
        layers: list[nn.Module] = []
        in_dim = d + 3  # token features + initial 3-D coordinate
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 3
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Identity-at-init: final layer produces a zero residual.
        final_linear = self.mlp[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        # Residual gate initialised at 0 so the head is identity at training start.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def _active_view_pool(
        self,
        tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool per-view ray tokens over active views to per-joint features.

        Args:
            tokens: ``(B, T, V, J, d)``.
            view_mask: ``(B, T, V)`` bool.

        Returns:
            pooled: ``(B, T, J, d)``.
        """
        B, T, V, J, d = tokens.shape
        mask = view_mask[:, :, :, None, None].float()  # (B, T, V, 1, 1)
        pooled = (tokens * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)  # (B, T, J, d)
        return pooled

    def forward(
        self,
        tokens: torch.Tensor,
        pred_3d_init: torch.Tensor,
        view_mask: torch.Tensor,
        active_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine the initial 3-D estimate for sparse-view inputs.

        Args:
            tokens: ``(B, T, V, J, d)`` ray tokens.
            pred_3d_init: ``(B, T, J, 3)`` initial triangulated estimate.
            view_mask: ``(B, T, V)`` bool.  True = view is active.
            active_count: optional ``(B, T)`` number of active views per frame.
                If None, it is computed from ``view_mask``.

        Returns:
            pred_3d_sparse: ``(B, T, J, 3)`` refined 3-D joints.
        """
        B, T, V, J, d = tokens.shape
        if d != self.d:
            raise ValueError(
                f"Token dimension {d} does not match head dimension {self.d}"
            )
        if view_mask.shape != (B, T, V):
            raise ValueError(
                f"view_mask shape {view_mask.shape} does not match tokens shape "
                f"{(B, T, V, J, d)}"
            )

        # Pool active-view tokens.
        pooled = self._active_view_pool(tokens, view_mask)  # (B, T, J, d)

        # Add count embedding if requested.
        if self.use_count_embedding and self.count_embed is not None:
            if active_count is None:
                active_count = view_mask.sum(dim=-1)  # (B, T)
            active_count = active_count.clamp(min=0, max=self.n_views).long()
            count_emb = self.count_embed(active_count)  # (B, T, d)
            pooled = pooled + count_emb[:, :, None, :]

        # Predict a residual correction around the initial DLT estimate.
        x = torch.cat([pooled, pred_3d_init], dim=-1)  # (B, T, J, d + 3)
        residual = self.mlp(x)  # (B, T, J, 3)
        gate = torch.tanh(self.residual_gate)
        return pred_3d_init + gate * residual

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"d={self.d}, n_views={self.n_views}, n_joints={self.n_joints}, "
            f"hidden={self.hidden}, n_layers={self.n_layers}, "
            f"use_count_embedding={self.use_count_embedding})"
        )


if __name__ == "__main__":
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d)
    pred_3d_init = torch.randn(B, T, J, 3)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    # Make one sample sparse (k=2).
    view_mask[0, :, 2:] = False

    head = SeparateSparseViewHeadV86(d=d, n_views=V, n_joints=J)
    out = head(tokens, pred_3d_init, view_mask)
    assert out.shape == (B, T, J, 3)
    # Identity-at-init: output should equal input up to numerical noise.
    assert torch.allclose(out, pred_3d_init, atol=1e-6)
    print("SeparateSparseViewHeadV86 CPU smoke test passed")
