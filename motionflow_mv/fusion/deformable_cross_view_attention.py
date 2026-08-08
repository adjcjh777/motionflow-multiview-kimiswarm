"""Deformable cross-view attention guided by epipolar geometry.

This module implements a sparse cross-view attention block.  For each query
view/joint token it samples only a small, learned subset of key views instead
of attending to every view.  Epipolar consistency between views provides the
geometric prior that guides the sparse sampler, while a content term lets the
network refine that selection.

The sampler is kept differentiable via a straight-through hard top-k mask on the
soft attention weights, so the module can be plugged into any end-to-end pose
estimation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.epipolar_attention_bias import compute_epipolar_distance


class DeformableCrossViewAttention(nn.Module):
    """Sparse, epipolar-geometry-aware cross-view attention.

    Parameters
    ----------
    d:
        Token dimension. Must be divisible by ``n_heads``.
    n_heads:
        Number of attention heads.
    n_views:
        Maximum number of views (used for validation only).
    n_samples:
        Number of key views sampled for each query view/joint.  ``1 <= n_samples <= V``.
    epipolar_temperature:
        Temperature for converting epipolar distances to additive logits.
    dropout:
        Dropout probability on the output projection.
    use_topk_straight_through:
        If ``True``, use a hard top-k mask in the forward pass and back-propagate
        through the re-normalised soft weights (straight-through estimator).  If
        ``False`` (default), keep the fully differentiable soft attention used by
        v25/v26 production runs.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_views: int = 4,
        n_samples: int = 2,
        epipolar_temperature: float = 10.0,
        dropout: float = 0.0,
        use_topk_straight_through: bool = False,
    ):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")

        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.n_samples = n_samples
        self.epipolar_temperature = epipolar_temperature
        self.use_topk_straight_through = use_topk_straight_through
        self.head_dim = d // n_heads

        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

        # Zero-initialized residual gate so the module starts as identity and
        # learns the cross-view contribution gradually.
        self.residual_scale = nn.Parameter(torch.zeros(1))

        # Learnable geometry gate.  Softplus keeps the contribution of epipolar
        # distance non-negative (larger distance -> smaller attention weight).
        self.geometry_scale = nn.Parameter(torch.tensor(1.0))

        # Straight-through temperature.  Kept as a parameter for inspection but
        # not used to control the hard top-k; it only affects the backward path
        # through the soft weights.
        self.tau = nn.Parameter(torch.tensor(0.5))

    def _prepare_view_mask(
        self,
        view_mask: torch.Tensor | None,
        B: int,
        T: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize ``view_mask`` to ``(B * T, V)`` float mask."""
        if view_mask is None:
            return torch.ones(B * T, V, device=device)
        if view_mask.dim() == 2:
            if view_mask.shape == (B, V):
                return view_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, V)
            if view_mask.shape == (B * T, V):
                return view_mask
            raise ValueError(
                f"view_mask (N, V) shape {view_mask.shape} incompatible with "
                f"B={B}, T={T}, V={V}"
            )
        if view_mask.dim() == 3:
            if view_mask.shape != (B, T, V):
                raise ValueError(
                    f"view_mask (B, T, V) shape {view_mask.shape} incompatible with "
                    f"B={B}, T={T}, V={V}"
                )
            return view_mask.reshape(B * T, V)
        raise ValueError(
            f"view_mask must be (B, T, V), (B, V) or (N, V), got {view_mask.shape}"
        )

    def forward(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        points_2d: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply deformable cross-view attention.

        Args
        ----
        x:
            ``(B, T, V, J, d)`` per-view per-joint feature tokens.
        K, R, t:
            Intrinsics ``(B*T, V, 3, 3)``, rotations ``(B*T, V, 3, 3)`` and
            translation ``(B*T, V, 3)``.
        points_2d:
            ``(B*T, V, J, 2)`` image points used to compute epipolar geometry.
        view_mask:
            Optional binary mask of shape ``(B, T, V)`` or ``(B, V)`` marking
            present (1) and missing (0) views.

        Returns
        -------
            ``(B, T, V, J, d)`` updated tokens.
        """
        if x.dim() != 5:
            raise ValueError(f"x must be (B, T, V, J, d), got shape {x.shape}")

        B, T, V, J, d = x.shape
        N = B * T
        device = x.device

        view_mask_flat = self._prepare_view_mask(view_mask, B, T, V, device)

        # Flatten batch and time.
        x_flat = x.reshape(N, V, J, d)

        # Q, K, V: (N, V, J, 3, H, Dh) -> separate.
        qkv = self.qkv(x_flat).reshape(N, V, J, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)  # each (N, V, J, H, Dh)

        # Content logits: scaled dot product over heads.
        # (N, H, V_q, V_k, J)
        content_logits = torch.einsum("nvjhd,nkjhd->nhvkj", q, k) * (self.head_dim ** -0.5)

        # Epipolar distance between every view pair for each joint.
        # Distances are symmetric in this implementation.
        dist = compute_epipolar_distance(K, R, t, points_2d)  # (N, V, V, J)

        # Geometry-aware logits.  Larger epipolar distance -> more negative logit.
        geometry_logits = -dist.unsqueeze(1) * F.softplus(self.geometry_scale) / self.epipolar_temperature
        logits = content_logits + geometry_logits  # (N, H, V_q, V_k, J)

        # Apply key-view mask: masked-out views cannot be attended to.
        if view_mask is not None:
            key_mask = view_mask_flat.view(N, 1, 1, V, 1)  # (N, 1, 1, V, 1)
            logits = logits.masked_fill(key_mask == 0, -1e9)

        # Soft attention weights over all key views (fully differentiable).
        weights = F.softmax(logits / F.softplus(self.tau).clamp_min(1e-3), dim=3)

        # Optional sparse top-k sampling with a straight-through estimator.
        # Forward uses a hard, re-normalised top-k mask; backward flows through
        # the re-normalised soft weights inside the selected subset.
        if self.use_topk_straight_through and V > 1:
            k = min(self.n_samples, V)
            if k < V:
                topk_idx = torch.topk(weights, k, dim=3).indices  # (N, H, V_q, k, J)
                hard_mask = torch.zeros_like(weights).scatter_(3, topk_idx, 1.0)
                # Forward: uniform average over selected key views.
                hard_weights = hard_mask / hard_mask.sum(dim=3, keepdim=True).clamp_min(1e-8)
                # Backward path: re-normalised soft weights inside the top-k set.
                masked_soft = weights * hard_mask
                denom = masked_soft.sum(dim=3, keepdim=True).clamp_min(1e-8)
                masked_soft = masked_soft / denom
                # Straight-through: forward uses hard_weights, backward uses masked_soft.
                sparse_weights = hard_weights + (masked_soft - masked_soft.detach())
            else:
                sparse_weights = weights
        else:
            sparse_weights = weights

        # Aggregate values: output shape (N, V_q, J, H, Dh).
        out = torch.einsum("nhvkj,nvjhd->nvjhd", sparse_weights, v)

        # Restore view dimension ordering and apply output projection.
        out = out.reshape(N, V, J, d)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Gated residual update: start as identity, learn the cross-view contribution.
        out = x_flat + self.residual_scale * out
        out = self.norm(out)

        # Zero out masked query views so they do not leak downstream.
        if view_mask is not None:
            query_mask = view_mask_flat.view(N, V, 1, 1)
            out = out * query_mask

        return out.view(B, T, V, J, d)


if __name__ == "__main__":
    B, T, V, J, d = 2, 3, 4, 17, 64
    x = torch.randn(B, T, V, J, d)
    N = B * T
    K = torch.eye(3).unsqueeze(0).expand(V, -1, -1).clone()
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    K = K.unsqueeze(0).expand(N, -1, -1, -1)
    R = torch.eye(3).unsqueeze(0).expand(N, V, -1, -1)
    t = torch.zeros(N, V, 3)
    points_2d = torch.randn(N, V, J, 2) * 100.0

    for use_topk in (False, True):
        module = DeformableCrossViewAttention(
            d=d, n_heads=4, n_views=V, n_samples=2, use_topk_straight_through=use_topk
        )
        out = module(x, K, R, t, points_2d)
        assert out.shape == x.shape
        loss = out.sum()
        loss.backward()
        assert any(p.grad is not None for p in module.parameters())
    print("DeformableCrossViewAttention CPU smoke test passed")
