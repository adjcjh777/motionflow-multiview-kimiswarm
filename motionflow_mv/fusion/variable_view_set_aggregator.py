"""Permutation-invariant set transformer for variable numbers of views.

The aggregator treats the multi-view tokens as an unordered set and applies
Induced Set Attention Blocks (ISABs).  Each ISAB uses a small set of inducing
points to reduce complexity from quadratic to linear in the number of views.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InducedSetAttentionBlock(nn.Module):
    """One Induced Set Attention Block (ISAB).

    Parameters
    ----------
    d:
        Token dimension.
    n_heads:
        Number of attention heads.
    num_inducing:
        Number of inducing points ``I``.
    dropout:
        Dropout probability for ``MultiheadAttention``.
    """

    def __init__(self, d: int, n_heads: int, num_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.num_inducing = num_inducing

        self.inducing_points = nn.Parameter(torch.randn(num_inducing, d) * 0.02)
        self.attn_enc = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dec = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one ISAB.

        Args
        ----
        x:
            ``(B, V, d)`` where ``V`` may vary across batches.

        Returns
        -------
        ``(B, V, d)`` updated tokens.
        """
        B, V, d = x.shape
        inducing = self.inducing_points.unsqueeze(0).expand(B, -1, -1)  # (B, I, d)

        # Encode: attend from inducing points to the input set.
        h, _ = self.attn_enc(inducing, x, x)  # (B, I, d)
        # Decode: attend from input set back to inducing points.
        out, _ = self.attn_dec(x, h, h)  # (B, V, d)
        return self.norm(x + out)


class VariableViewSetAggregator(nn.Module):
    """Permutation-invariant aggregator over an arbitrary set of views.

    Parameters
    ----------
    d:
        Token dimension.
    n_heads:
        Attention heads inside each ISAB.
    n_isab_layers:
        Number of stacked ISAB layers.
    num_inducing_points:
        Number of inducing points in each ISAB.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_isab_layers: int = 2,
        num_inducing_points: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.layers = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    d=d,
                    n_heads=n_heads,
                    num_inducing=num_inducing_points,
                    dropout=dropout,
                )
                for _ in range(n_isab_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate view tokens.

        Args
        ----
        x:
            ``(B, T, V, J, d)`` tokens.

        Returns
        -------
        Aggregated tokens of the same shape ``(B, T, V, J, d)``.  The operation is
        permutation-equivariant over the ``V`` dimension.
        """
        B, T, V, J, d = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B, V, J * T, d)
        x = x.permute(0, 2, 1, 3).reshape(B * J * T, V, d)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, J * T, V, d).permute(0, 2, 1, 3).reshape(B, V, T, J, d)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, V, J, d)
        return x


if __name__ == "__main__":
    B, T, V, J, d = 2, 5, 4, 17, 64
    x = torch.randn(B, T, V, J, d)

    agg = VariableViewSetAggregator(
        d=d,
        n_heads=4,
        n_isab_layers=2,
        num_inducing_points=32,
        dropout=0.0,
    )
    out = agg(x)
    assert out.shape == (B, T, V, J, d)

    # Permutation equivariance: permuting views should permute outputs identically.
    perm = torch.randperm(V)
    x_perm = x[:, :, perm, :, :]
    out_perm = agg(x_perm)
    # Small floating-point drift is expected across unrelated MHA calls; the
    # outputs are permutation-equivariant up to numerical precision.
    assert torch.allclose(out[:, :, perm, :, :], out_perm, atol=1e-6, rtol=1e-4)

    # Gradient sanity.
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in agg.parameters())
    print("VariableViewSetAggregator CPU smoke test passed")
