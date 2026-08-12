"""v85: Random View Dropout with view-count conditioning.

A lightweight training-time augmentation that randomly drops entire camera
views and injects a learned embedding of the active-view count into the ray
tokens.  The goal is to make the downstream triangulation head robust to
sparse-view inputs (k=2, k=3) without relying on a learned uncertainty
estimator.

At evaluation time no views are dropped; the count embedding simply tells the
model how many views are present.

The module is identity-at-init: the count embedding is zero-initialised, so
with ``dropout_prob=0`` it does not change the v25 baseline.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class RandomViewDropoutV85(nn.Module):
    """Random whole-view dropout plus active-view-count token embedding.

    Parameters
    ----------
    d:
        Token dimension (must match the v25 ray-token dimension).
    n_views:
        Maximum number of camera views.
    dropout_prob:
        Probability of dropping each view during training.  The actual drop is
        sampled independently per view, then forced to keep at least
        ``min_views`` active views.
    min_views:
        Minimum number of views to retain after dropout.
    use_count_embedding:
        If True, add a learned embedding keyed by the number of active views.
    """

    def __init__(
        self,
        d: int = 128,
        n_views: int = 4,
        dropout_prob: float = 0.3,
        min_views: int = 2,
        use_count_embedding: bool = True,
    ):
        super().__init__()
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
        if min_views < 1:
            raise ValueError(f"min_views must be >= 1, got {min_views}")
        if min_views > n_views:
            raise ValueError(
                f"min_views ({min_views}) cannot exceed n_views ({n_views})"
            )

        self.d = d
        self.n_views = n_views
        self.dropout_prob = dropout_prob
        self.min_views = min_views
        self.use_count_embedding = use_count_embedding

        if use_count_embedding:
            # Embed the active view count in [0, n_views].
            self.count_embed = nn.Embedding(n_views + 1, d)
            nn.init.zeros_(self.count_embed.weight)
        else:
            self.count_embed = None

    def apply_dropout(
        self,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a view mask with additional views randomly dropped.

        Args:
            view_mask: ``(B, T, V)`` bool. ``True`` / ``1`` means the view is
                originally available.  If ``None``, all views are assumed valid.

        Returns:
            new_view_mask: ``(B, T, V)`` bool mask after dropout.
        """
        if not self.training or self.dropout_prob <= 0.0:
            if view_mask is None:
                raise ValueError("view_mask must be provided when dropout is disabled")
            return view_mask

        if view_mask is None:
            raise ValueError("view_mask must be provided when dropout is enabled")

        B, T, V = view_mask.shape
        device = view_mask.device
        dtype = torch.float32

        # Base keep probability per view.
        keep_prob = 1.0 - self.dropout_prob
        keep = torch.bernoulli(torch.full((B, T, V), keep_prob, device=device, dtype=dtype))

        # Respect the original mask.
        keep = keep * view_mask.float()

        # Ensure at least min_views active per (B, T).
        keep = self._enforce_min_views(keep, view_mask)

        return keep.bool()

    def _enforce_min_views(
        self,
        keep: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Force each sample to keep at least ``min_views`` views.

        Args:
            keep: ``(B, T, V)`` float in ``{0, 1}``.
            view_mask: ``(B, T, V)`` bool original mask.

        Returns:
            keep: ``(B, T, V)`` float with at least ``min_views`` ones.
        """
        B, T, V = keep.shape
        for b in range(B):
            for t in range(T):
                active = int(keep[b, t].sum().item())
                if active >= self.min_views:
                    continue
                needed = self.min_views - active
                # Candidate views are those that are valid but currently dropped.
                candidates = []
                for v in range(V):
                    if view_mask[b, t, v] and keep[b, t, v] == 0:
                        candidates.append(v)
                if not candidates:
                    continue
                # Deterministic sampling via torch.randperm keeps behaviour
                # reproducible when combined with the random keep sample above.
                perm = torch.randperm(len(candidates), device=keep.device)
                for i in range(min(needed, len(perm))):
                    keep[b, t, candidates[perm[i].item()]] = 1.0
        return keep

    def embed_tokens(
        self,
        tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add active-view-count embedding to tokens.

        Args:
            tokens: ``(B, T, V, J, d)``.
            view_mask: ``(B, T, V)`` bool.

        Returns:
            tokens: ``(B, T, V, J, d)`` with count embedding added.
        """
        if not self.use_count_embedding or self.count_embed is None:
            return tokens

        B, T, V, J, d = tokens.shape
        count = view_mask.sum(dim=-1)  # (B, T)
        count = count.clamp(min=0, max=V).long()
        emb = self.count_embed(count)  # (B, T, d)
        # Broadcast over views and joints.
        return tokens + emb[:, :, None, None, :]

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply view-count embedding only (dropout is handled via ``apply_dropout``).

        This forward is kept for API symmetry; it simply calls
        :meth:`embed_tokens`.
        """
        return self.embed_tokens(tokens, view_mask)


if __name__ == "__main__":
    B, T, V, J, d = 2, 4, 4, 17, 128
    tokens = torch.randn(B, T, V, J, d)
    view_mask = torch.ones(B, T, V, dtype=torch.bool)
    # Drop one view deterministically for testing.
    view_mask[:, :, -1] = False

    module = RandomViewDropoutV85(d=d, n_views=V, dropout_prob=0.5, min_views=2)
    module.train()
    dropped_mask = module.apply_dropout(view_mask)
    print("dropped mask shape:", dropped_mask.shape)
    print("active views per sample:", dropped_mask.sum(dim=-1))

    tokens_out = module(tokens, dropped_mask)
    assert tokens_out.shape == tokens.shape
    print("RandomViewDropoutV85 CPU smoke test passed")
