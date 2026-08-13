"""v87: Sparse-View Residual Head.

A lightweight dedicated head for variable-view inputs where the number of active
views ``k`` is smaller than the maximum ``n_views``.  It pools the per-view ray
tokens from the active cameras, injects a learned active-view-count embedding,
and predicts a residual 3-D pose correction around the initial DLT estimate.

Compared with the v86 separate sparse-view head, v87 uses a *per-view residual*
design: each active view predicts a joint-level residual, and a small attention
mechanism weights the contributions before the final correction.  This lets
different views contribute differently per joint while keeping the full-view
path untouched.

The module is identity-at-init: the final output layer is zero-initialised and
the scalar residual gate starts at zero, so the sparse-view branch has no effect
at the beginning of training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseViewResidualHeadV87(nn.Module):
    """Dedicated residual head for sparse-view (``k < n_views``) inputs.

    Parameters
    ----------
    d:
        Ray-token dimension (must match the v25 ray-token dimension).
    n_views:
        Maximum number of camera views (used to size the count embedding).
    n_joints:
        Number of body joints.
    hidden:
        Hidden dimension of the per-view and residual MLPs.
    n_layers:
        Number of layers in the per-view residual MLP (>= 1).
    dropout:
        Dropout probability inside the MLPs.
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

        # Per-view residual MLP: maps token + initial 3-D estimate to a 3-D
        # per-view residual per joint.
        view_layers: list[nn.Module] = []
        in_dim = d + 3
        for i in range(n_layers):
            is_last = i == n_layers - 1
            out_dim = 3 if is_last else hidden
            view_layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                view_layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    view_layers.append(nn.Dropout(dropout))
            in_dim = out_dim if not is_last else hidden
        self.view_residual_mlp = nn.Sequential(*view_layers)

        # Zero-initialize the final layer so the per-view residual is 0 at init.
        final_linear = self.view_residual_mlp[-1]
        assert isinstance(final_linear, nn.Linear)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        # Per-view attention weight: each view gets a scalar per joint based on
        # its token and the initial 3-D estimate.
        self.view_attention_mlp = nn.Sequential(
            nn.Linear(d + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

        # Layer norm on pooled per-joint features for stability.
        self.feature_norm = nn.LayerNorm(d)

        # Final residual gate, identity-at-init.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def _active_view_pool(
        self,
        tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool active-view tokens to per-joint features.

        Args:
            tokens: ``(B, T, V, J, d)``.
            view_mask: ``(B, T, V)`` bool.

        Returns:
            pooled: ``(B, T, J, d)``.
        """
        mask = view_mask[:, :, :, None, None].float()  # (B, T, V, 1, 1)
        pooled = (tokens * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return self.feature_norm(pooled)

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

        # Add count embedding to tokens if requested.
        if self.use_count_embedding and self.count_embed is not None:
            if active_count is None:
                count = view_mask.sum(dim=-1).clamp(min=0, max=self.n_views).long()
            else:
                count = active_count.clamp(min=0, max=self.n_views).long()
            emb = self.count_embed(count)  # (B, T, d)
            tokens = tokens + emb[:, :, None, None, :]

        # Per-view residual predictions.
        pred_3d_init_expanded = pred_3d_init.unsqueeze(2).expand(-1, -1, V, -1, -1)
        # tokens: (B, T, V, J, d); pred: (B, T, V, J, 3)
        view_input = torch.cat([tokens, pred_3d_init_expanded], dim=-1)
        view_residual = self.view_residual_mlp(view_input)  # (B, T, V, J, 3)

        # Per-view attention weights over active views.
        view_attn = self.view_attention_mlp(view_input).squeeze(-1)  # (B, T, V, J)
        # Mask out inactive views and apply softmax over the active ones.
        mask = view_mask[:, :, :, None].expand(-1, -1, -1, J)  # (B, T, V, J)
        view_attn = view_attn.masked_fill(~mask, float("-inf"))
        # Guard against all-inactive rows (should not happen with valid masks).
        view_attn = torch.where(
            torch.isinf(view_attn).all(dim=2, keepdim=True),
            torch.zeros_like(view_attn),
            view_attn,
        )
        view_weights = F.softmax(view_attn, dim=2)  # (B, T, V, J)
        view_weights = torch.nan_to_num(view_weights, nan=0.0)

        # Weighted combination of per-view residuals.
        weighted_residual = (view_weights.unsqueeze(-1) * view_residual).sum(dim=2)

        # Gated residual update around the initial DLT estimate.
        gate = torch.tanh(self.residual_gate)
        return pred_3d_init + gate * weighted_residual

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

    head = SparseViewResidualHeadV87(d=d, n_views=V, n_joints=J)
    out = head(tokens, pred_3d_init, view_mask)
    assert out.shape == (B, T, J, 3)
    # Identity-at-init: output should equal input up to numerical noise.
    assert torch.allclose(out, pred_3d_init, atol=1e-6)
    print("SparseViewResidualHeadV87 CPU smoke test passed")
