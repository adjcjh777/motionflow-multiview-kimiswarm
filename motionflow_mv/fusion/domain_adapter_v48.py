"""v48: Domain-adaptive feature modulation for cross-dataset generalization.

This module provides a lightweight, domain-conditional adapter that can be
inserted between blocks of ``OmniMultiViewFusionV5``.  It supports two
complementary adaptation mechanisms:

1. **FiLM (Feature-wise Linear Modulation)** — a domain embedding predicts
   per-channel affine parameters (gamma/beta) that modulate the incoming
   features.
2. **Conditional Batch Normalization** — per-domain affine parameters are
   applied after normalizing features across their spatial dimensions.

Both mechanisms are initialized so that the adapter is approximately an
identity mapping at the start of training, which preserves the behaviour of a
warm-started backbone.

An optional gradient-reversal (GRL) domain discriminator can be attached to
encourage domain-invariant representations.  The discriminator is trained to
predict the dataset/domain label while the GRL makes the upstream features
fool the discriminator.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    """Gradient-reversal helper for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # noqa: D401
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class _GradientReversalLayer(nn.Module):
    """Wrap the gradient-reversal function with a configurable ``lambda_``."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


class _FiLM(nn.Module):
    """Domain-conditional Feature-wise Linear Modulation.

    A domain embedding is projected through a small MLP to produce per-channel
    ``gamma`` and ``beta``.  The modulation is ``x * (1 + gamma) + beta`` so
    that zero-initialized MLP outputs give the identity mapping at the start of
    training.
    """

    def __init__(self, in_channels: int, num_domains: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_domains = num_domains

        self.domain_embed = nn.Embedding(num_domains, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels * 2),
        )
        # Zero-initialized last layer -> identity modulation at init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = self.domain_embed(domain_ids)  # (B, hidden)
        params = self.mlp(emb)  # (B, C * 2)
        gamma, beta = params.chunk(2, dim=-1)  # (B, C) each

        # Broadcast over the spatial dimensions (T, V, J).
        view_shape = [B] + [1] * (x.dim() - 2) + [self.in_channels]
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)

        return x * (1.0 + gamma) + beta


class _ConditionalBatchNorm(nn.Module):
    """Lightweight conditional batch normalization.

    Features are normalized using the mean and variance of the current batch,
    then scaled and shifted by per-domain learnable affine parameters.  This
    is shape-agnostic for tensors whose last dimension is the channel axis.
    """

    def __init__(self, in_channels: int, num_domains: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_domains = num_domains
        # Per-domain affine parameters.
        self.gamma = nn.Parameter(torch.ones(num_domains, in_channels))
        self.beta = nn.Parameter(torch.zeros(num_domains, in_channels))

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        # Normalize across all non-batch, non-channel dimensions.
        dims = list(range(1, x.dim() - 1))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) * torch.rsqrt(var + 1e-5)

        # Select and broadcast per-domain affine parameters.
        gamma = self.gamma[domain_ids]
        beta = self.beta[domain_ids]
        view_shape = [x.shape[0]] + [1] * (x.dim() - 2) + [self.in_channels]
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)

        return gamma * x_norm + beta


class _DomainDiscriminator(nn.Module):
    """Small MLP that predicts domain labels from pooled features."""

    def __init__(self, in_channels: int, num_domains: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # ``feat`` is expected to be a pooled (B, in_channels) vector.
        return self.net(feat)


class DomainAdapterV48(nn.Module):
    """Domain adapter combining FiLM, conditional BN, and an optional GRL discriminator.

    Parameters
    ----------
    in_channels:
        Number of channels/features in the input tensor (last dimension).
    num_domains:
        Number of distinct domains/datasets.  Default follows the v48 proposal
        (h36m, mpi, aist, shelf, campus, 3dpw).
    hidden:
        Hidden dimension of the domain embedding MLP.
    dropout:
        Dropout probability inside the FiLM and discriminator MLPs.
    use_film:
        If ``True``, apply domain-conditional FiLM modulation.
    use_conditional_bn:
        If ``True``, apply per-domain conditional batch normalization.
    use_grl_discriminator:
        If ``True``, attach a gradient-reversal domain discriminator on pooled
        features.
    grl_lambda:
        Scaling factor for the gradient reversal layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_domains: int = 6,
        hidden: int = 64,
        dropout: float = 0.1,
        use_film: bool = True,
        use_conditional_bn: bool = False,
        use_grl_discriminator: bool = True,
        grl_lambda: float = 0.1,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_domains <= 0:
            raise ValueError(f"num_domains must be positive, got {num_domains}")

        self.in_channels = in_channels
        self.num_domains = num_domains
        self.use_film = use_film
        self.use_conditional_bn = use_conditional_bn
        self.use_grl_discriminator = use_grl_discriminator

        if use_film:
            self.film = _FiLM(in_channels, num_domains, hidden, dropout=dropout)
        else:
            self.film = None

        if use_conditional_bn:
            self.conditional_bn = _ConditionalBatchNorm(in_channels, num_domains)
        else:
            self.conditional_bn = None

        if use_grl_discriminator:
            self.grl = _GradientReversalLayer(lambda_=grl_lambda)
            self.domain_discriminator = _DomainDiscriminator(
                in_channels, num_domains, hidden, dropout=dropout
            )
        else:
            self.grl = None
            self.domain_discriminator = None

    def _check_domain_ids(self, domain_ids: torch.Tensor) -> None:
        if domain_ids.dim() != 1:
            raise ValueError(f"domain_ids must be 1D, got shape {domain_ids.shape}")
        if domain_ids.numel() == 0:
            raise ValueError("domain_ids must not be empty")
        min_id = domain_ids.min().item()
        max_id = domain_ids.max().item()
        if min_id < 0 or max_id >= self.num_domains:
            raise ValueError(
                f"domain_ids must be in [0, {self.num_domains - 1}], "
                f"got min={min_id}, max={max_id}"
            )

    def forward(
        self,
        feat: torch.Tensor,
        dataset_id: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply domain-conditional adaptation.

        Args
        ----
        feat:
            ``(B, T, V, J, C)`` multi-view feature tensor.  The last dimension
            must equal ``in_channels``.
        dataset_id:
            ``(B,)`` integer domain ids in ``[0, num_domains)``.
        view_mask:
            ``(B, T, V)`` optional binary mask.  Reserved for downstream use;
            currently not used by the adapter itself.

        Returns
        -------
        adapted_feat:
            ``(B, T, V, J, C)`` modulated features.
        domain_logits:
            ``(B, num_domains)`` logits from the optional GRL domain
            discriminator, or ``None`` if ``use_grl_discriminator=False``.
        """
        if feat.dim() < 2:
            raise ValueError(f"feat must be at least 2D, got {feat.dim()}D tensor")
        if feat.shape[-1] != self.in_channels:
            raise ValueError(
                f"feat last dimension ({feat.shape[-1]}) must match "
                f"in_channels ({self.in_channels})"
            )

        # view_mask is part of the public API but unused by this module.
        _ = view_mask

        self._check_domain_ids(dataset_id)

        out = feat
        if self.conditional_bn is not None:
            out = self.conditional_bn(out, dataset_id)
        if self.film is not None:
            out = self.film(out, dataset_id)

        domain_logits: Optional[torch.Tensor] = None
        if self.use_grl_discriminator and self.domain_discriminator is not None:
            # Apply GRL before pooling so the discriminator gradients flow back
            # with reversed sign to the rest of the network.
            pooled = out.mean(dim=list(range(1, out.dim() - 1)))
            domain_logits = self.domain_discriminator(self.grl(pooled))

        return out, domain_logits


if __name__ == "__main__":
    B, T, V, J, C = 2, 5, 4, 17, 32
    feat = torch.randn(B, T, V, J, C)
    dataset_id = torch.tensor([0, 5], dtype=torch.long)

    # Full adapter.
    adapter = DomainAdapterV48(
        in_channels=C,
        num_domains=6,
        hidden=64,
        use_film=True,
        use_conditional_bn=True,
        use_grl_discriminator=True,
    )
    out, logits = adapter(feat, dataset_id)
    assert out.shape == feat.shape
    assert logits is not None
    assert logits.shape == (B, 6)

    # Identity-like behaviour at init when only FiLM is enabled.
    adapter_film_only = DomainAdapterV48(
        in_channels=C,
        num_domains=6,
        use_film=True,
        use_conditional_bn=False,
        use_grl_discriminator=False,
    )
    out_film, logits_film = adapter_film_only(feat, dataset_id)
    assert torch.allclose(out_film, feat, atol=1e-5)
    assert logits_film is None

    # Gradient sanity.
    loss = out.sum() + (logits.sum() if logits is not None else 0.0)
    loss.backward()
    assert any(p.grad is not None for p in adapter.parameters())

    print("DomainAdapterV48 CPU smoke test passed")
