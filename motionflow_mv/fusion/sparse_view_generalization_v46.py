"""v46: Sparse-View Generalization (SVG) for MotionFlow-MultiView.

This module adds a lightweight, view-agnostic reliability head that predicts
per-view weights from multi-view feature tokens.  It is designed to be dropped
into ``OmniMultiViewFusionV5`` downstream of the v25 geometry fusion block; the
predicted weights can then be fed back into the weighted DLT triangulation step
so that sparse or dropped views contribute appropriately.

The module keeps an identity-like behaviour at initialization: every view is
given a weight close to one when training starts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .variable_view_set_aggregator import VariableViewSetAggregator


class SparseViewGeneralizationV46(nn.Module):
    """Predict per-view reliability weights for sparse/variable-view fusion.

    Parameters
    ----------
    in_channels:
        Dimension of the incoming multi-view feature tokens.
    n_views:
        Number of camera views (kept for API compatibility; the module itself
        treats the view dimension as an unordered set).
    hidden:
        Hidden dimension of the per-view reliability MLP.
    dropout:
        Dropout probability applied inside the ISAB set aggregator.
    """

    def __init__(
        self,
        in_channels: int,
        n_views: int,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_views = n_views
        self.hidden = hidden

        # View-agnostic set aggregator: handles variable V via inducing points.
        self.set_aggregator = VariableViewSetAggregator(
            d=in_channels,
            n_heads=4,
            n_isab_layers=2,
            num_inducing_points=32,
            dropout=dropout,
        )

        # Per-view reliability head over pooled per-view features.
        self.reliability_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Zero-initialized final layer gives sigmoid(0) = 0.5, i.e. weight ~1.0.
        with torch.no_grad():
            final_linear = self.reliability_mlp[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-view reliability weights.

        Args
        ----
        x:
            ``(B, T, V, J, C)`` multi-view feature tokens.
        view_mask:
            Optional binary mask of shape ``(B, T, V)`` or ``(B, V)``.
            A value of ``0`` / ``False`` means the view is absent.

        Returns
        -------
        reliability:
            ``(B, T, V, J)`` positive weights, zero for masked-out views.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected x to be 5D (B, T, V, J, C), got {x.dim()}D")

        B, T, V, J, C = x.shape

        # Aggregate features across the view set.
        x = self.set_aggregator(x, view_mask=view_mask)  # (B, T, V, J, C)

        # Pool per-view features: mean and std over joints.
        mean_feat = x.mean(dim=3)  # (B, T, V, C)
        std_feat = x.std(dim=3, unbiased=False)
        feat = torch.cat([mean_feat, std_feat], dim=-1)  # (B, T, V, 2*C)

        # Per-view reliability in (0, 2), approximately 1.0 at init.
        r = self.reliability_mlp(feat).squeeze(-1)  # (B, T, V)
        reliability = 2.0 * torch.sigmoid(r)

        # Broadcast to per-joint weights.
        reliability = reliability.unsqueeze(-1).expand(-1, -1, -1, J)  # (B, T, V, J)

        # Apply view mask.
        if view_mask is not None:
            mask = view_mask.float().unsqueeze(-1)  # (B, T, V, 1)
            if mask.dim() == 4:
                # view_mask was (B, T, V) -> mask (B, T, V, 1)
                pass
            elif mask.dim() == 3:
                # view_mask was (B, V) -> mask (B, V, 1); expand over T.
                mask = mask.unsqueeze(1).expand(-1, T, -1, -1)
            reliability = reliability * mask

        return reliability


if __name__ == "__main__":
    B, T, V, J, C = 2, 5, 4, 17, 64
    x = torch.randn(B, T, V, J, C)

    module = SparseViewGeneralizationV46(in_channels=C, n_views=V, hidden=32)

    # Full views.
    out = module(x)
    assert out.shape == (B, T, V, J)
    assert out.min().item() > 0.0
    # Identity-like at init: weights near 1.0.
    assert torch.allclose(out, torch.ones_like(out), atol=0.2)

    # Sparse views.
    view_mask = torch.ones(B, T, V)
    view_mask[:, :, -1] = 0.0
    out_masked = module(x, view_mask=view_mask)
    assert out_masked[..., :-1, :].max().item() > 0.0
    assert out_masked[..., -1, :].max().item() == 0.0

    # Gradient sanity.
    loss = out_masked.sum()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())

    print("SparseViewGeneralizationV46 CPU smoke test passed")
