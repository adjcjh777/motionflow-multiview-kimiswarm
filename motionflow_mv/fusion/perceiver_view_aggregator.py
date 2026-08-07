"""Perceiver-style permutation-invariant view aggregation.

Replaces the ISAB-based VariableViewSetAggregator with a small Perceiver that
cross-attends a fixed set of latent vectors to the variable number of views,
then decodes back to per-view tokens.  This usually gives better gradient flow
and stronger pooling than inducing-point attention, while still handling
arbitrary view subsets through ``view_mask``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverViewAggregator(nn.Module):
    """Perceiver aggregator over an arbitrary set of views.

    Parameters
    ----------
    d:
        Token dimension.
    n_heads:
        Number of attention heads.
    n_latents:
        Number of latent vectors used to summarise the view set.
    n_layers:
        Number of latent-block layers.  Each layer performs a latent-self-attention
        followed by a cross-attention to the views and a feed-forward.
    dropout:
        Dropout probability for attention layers.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_latents: int = 16,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_latents = n_latents
        self.n_layers = n_layers

        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)

        self.latent_self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.cross_to_latent = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.cross_to_views = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.latent_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.ReLU(),
                    nn.Linear(d * 2, d),
                )
                for _ in range(n_layers)
            ]
        )
        self.view_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.ReLU(),
                    nn.Linear(d * 2, d),
                )
                for _ in range(n_layers)
            ]
        )

        self.norm_latent_self = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_layers)]
        )
        self.norm_latent_cross = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_layers)]
        )
        self.norm_view_cross = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate view tokens.

        Args
        ----
        x:
            ``(B, T, V, J, d)`` tokens.
        view_mask:
            Optional binary mask of shape ``(B, T, V)`` or ``(B, V)``.
            A value of ``0`` means the view is absent for that sample/time.

        Returns
        -------
            ``(B, T, V, J, d)`` aggregated tokens.  Masked-out views are zeroed.
        """
        B, T, V, J, d = x.shape
        # Flatten batch/time/joint into one dimension; each slice is a set of V views.
        x_flat = x.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)

        # Expand latents for each set element.
        latents = self.latents.unsqueeze(0).expand(B * T * J, -1, -1)

        # Prepare key-padding mask for views: ``(N, V)`` with True for masked views.
        key_mask: torch.Tensor | None = None
        if view_mask is not None:
            if view_mask.dim() == 2:
                # (B, V) -> expand over T and J
                key_mask = (
                    view_mask.unsqueeze(1)
                    .expand(-1, T, -1)
                    .reshape(B * T, V)
                    .unsqueeze(1)
                    .expand(-1, J, -1)
                    .reshape(B * T * J, V)
                    .bool()
                )
            elif view_mask.dim() == 3:
                # (B, T, V)
                key_mask = (
                    view_mask.reshape(B * T, V)
                    .unsqueeze(1)
                    .expand(-1, J, -1)
                    .reshape(B * T * J, V)
                    .bool()
                )
            else:
                raise ValueError(
                    f"view_mask must be (B, T, V) or (B, V), got {view_mask.shape}"
                )
            key_mask = ~key_mask  # True where view is absent

        for i in range(self.n_layers):
            # 1. Latent self-attention.
            latents2, _ = self.latent_self_attn[i](latents, latents, latents)
            latents = self.norm_latent_self[i](latents + latents2)

            # 2. Cross-attention: latents query the view set.
            update, _ = self.cross_to_latent[i](
                latents,
                x_flat,
                x_flat,
                key_padding_mask=key_mask,
            )
            latents = self.norm_latent_cross[i](latents + update)
            latents = latents + self.latent_ffn[i](latents)

            # 3. Cross-attention: views query the latent summary.
            update, _ = self.cross_to_views[i](
                x_flat,
                latents,
                latents,
            )
            x_flat = self.norm_view_cross[i](x_flat + update)
            x_flat = x_flat + self.view_ffn[i](x_flat)

        # Zero-out masked views so they cannot leak downstream.
        if key_mask is not None:
            x_flat = x_flat.masked_fill(key_mask.unsqueeze(-1), 0.0)

        # Restore shape.
        x_out = x_flat.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)
        return x_out


if __name__ == "__main__":
    B, T, V, J, d = 2, 5, 4, 17, 64
    x = torch.randn(B, T, V, J, d)
    agg = PerceiverViewAggregator(d=d, n_heads=4, n_latents=16, n_layers=2)
    out = agg(x)
    assert out.shape == (B, T, V, J, d)

    # Permutation equivariance up to numerical precision.
    perm = torch.randperm(V)
    x_perm = x[:, :, perm, :, :]
    out_perm = agg(x_perm)
    assert torch.allclose(out[:, :, perm, :, :], out_perm, atol=1e-5, rtol=1e-4)

    # Masking: view 2 and 3 should not contribute.
    view_mask = torch.zeros(B, T, V)
    view_mask[:, :, :2] = 1.0
    out_masked = agg(x, view_mask=view_mask)
    assert out_masked[:, :, 2:, :].abs().max().item() < 1e-6

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in agg.parameters())
    print("PerceiverViewAggregator CPU smoke test passed")
