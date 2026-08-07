"""Cross-view transformer v17 — camera/ray-embedding aware view aggregation.

Replaces or augments the permutation-invariant set-view aggregator in
``OmniMultiViewFusionV5`` with a transformer that explicitly attends across
views using geometric embeddings derived from calibrated cameras and 2D
keypoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _CrossViewTransformerLayer(nn.Module):
    """Single cross-view transformer encoder layer.

    Queries and keys are biased by a per-view ray/camera embedding so that the
    attention pattern can depend on the relative geometry of the multi-view
    rig.  Values are not biased, keeping the representation rooted in image
    features.
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        ray_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one cross-view transformer layer.

        Args
        ----
        x:
            ``(N*J, V, d)`` per-view tokens.
        ray_emb:
            ``(N*J, V, d)`` geometric embedding for each view token.
        key_padding_mask:
            ``(N*J, V)`` boolean mask; ``True`` marks a view to be ignored.

        Returns
        -------
        ``(N*J, V, d)`` updated tokens.
        """
        # Bias queries and keys by the geometric embedding.
        q = self.q_proj(x) + ray_emb
        k = self.k_proj(x) + ray_emb
        v = self.v_proj(x)

        attn_out, _ = self.attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class CrossViewTransformerV17(nn.Module):
    """Cross-view transformer encoder with ray/camera embeddings.

    Parameters
    ----------
    d:
        Token dimension.
    n_heads:
        Number of attention heads per layer.
    n_layers:
        Number of stacked cross-view transformer layers.
    dropout:
        Dropout probability.

    Notes
    -----
    The module expects inputs ``x`` of shape ``(B, T, V, J, d)`` and optional
    camera parameters ``K, R, t`` together with 2D keypoints ``points_2d`` of
    shape ``(B, T, V, J, 2)``.  When any of the geometric inputs is missing, the
    module falls back to a standard transformer over views with no geometric
    embedding.
    """

    def __init__(self, d: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                _CrossViewTransformerLayer(d=d, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        # Project concatenated world ray direction and camera centre to ``d``.
        self.ray_proj = nn.Linear(6, d)

    def _build_ray_embedding(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        points_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Build a per-view, per-joint ray/camera embedding.

        Args
        ----
        K:
            ``(B*T, V, 3, 3)`` intrinsics.
        R:
            ``(B*T, V, 3, 3)`` extrinsic rotations (world -> camera).
        t:
            ``(B*T, V, 3)`` extrinsic translations (world -> camera).
        points_2d:
            ``(B*T, V, J, 2)`` 2D keypoints in image coordinates.

        Returns
        -------
        ``(B*T, V, J, d)`` ray/camera embedding.  If ``points_2d`` is passed as
        ``(B, T, V, J, 2)`` it is reshaped internally to ``(B*T, V, J, 2)``.
        """
        input_5d = points_2d.dim() == 5
        if input_5d:
            B, T, V, J, _ = points_2d.shape
            points_2d = points_2d.reshape(B * T, V, J, 2)
        else:
            B = None
            T = None
            V = None
            J = None

        N, V, J, _ = points_2d.shape

        # Avoid singular intrinsics by adding a small ridge if needed.
        K_inv = torch.linalg.inv(
            K
            + 1e-6
            * torch.eye(3, device=K.device, dtype=K.dtype).view(1, 1, 3, 3)
        )

        # Homogeneous image points.
        ones = torch.ones(N, V, J, 1, device=points_2d.device, dtype=points_2d.dtype)
        uv1 = torch.cat([points_2d, ones], dim=-1)  # (N, V, J, 3)

        # Direction in camera coordinates: K^{-1} * [u, v, 1].
        dir_cam = (K_inv.unsqueeze(2) @ uv1.unsqueeze(-1)).squeeze(-1)  # (N, V, J, 3)

        # Direction in world coordinates: R^T * dir_cam, then normalise.
        R_world = R.transpose(-2, -1)  # (N, V, 3, 3)
        world_dir = (R_world.unsqueeze(2) @ dir_cam.unsqueeze(-1)).squeeze(-1)
        world_dir = world_dir / (world_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # Camera centre in world coordinates: c = -R^T * t.
        cam_center = -(R_world @ t.unsqueeze(-1)).squeeze(-1)  # (N, V, 3)
        cam_center = cam_center.unsqueeze(2).expand(-1, -1, J, -1)  # (N, V, J, 3)

        geom = torch.cat([world_dir, cam_center], dim=-1)  # (N, V, J, 6)
        ray_emb = self.ray_proj(geom)  # (N, V, J, d)
        if input_5d:
            ray_emb = ray_emb.view(B, T, V, J, self.d)
        return ray_emb

    def _prepare_key_padding_mask(
        self,
        view_mask: torch.Tensor,
        B: int,
        T: int,
        V: int,
        J: int,
    ) -> torch.Tensor:
        """Convert a view mask to an MHA key-padding mask.

        ``True`` in the returned mask means "ignore this view".
        """
        if view_mask.dim() == 2:
            # (B, V) or (B*T, V)
            if view_mask.shape[0] == B and view_mask.shape[1] == V:
                key_mask = view_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, V)
            else:
                key_mask = view_mask.reshape(B * T, V)
        elif view_mask.dim() == 3:
            # (B, T, V)
            key_mask = view_mask.reshape(B * T, V)
        else:
            raise ValueError(
                f"view_mask must be (B, T, V) or (B, V), got {view_mask.shape}"
            )

        # Repeat across joint dimension; shape (B*T*J, V).
        key_mask = key_mask.unsqueeze(1).expand(-1, J, -1).reshape(B * T * J, V)
        return ~key_mask.bool()  # True where view is absent

    def _reshape_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T, V, J, d)`` -> ``(B*T*J, V, d)``."""
        B, T, V, J, d = x.shape
        return x.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)

    def _reshape_from_tokens(
        self, x: torch.Tensor, B: int, T: int, V: int, J: int, d: int
    ) -> torch.Tensor:
        """Reshape ``(B*T*J, V, d)`` -> ``(B, T, V, J, d)``."""
        return x.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)

    def forward(
        self,
        x: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        points_2d: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-view attention using ray/camera embeddings.

        Args
        ----
        x:
            ``(B, T, V, J, d)`` per-view feature tokens.
        K, R, t:
            Camera intrinsics/extrinsics. ``K`` and ``R`` have shape
            ``(B*T, V, 3, 3)`` and ``t`` has shape ``(B*T, V, 3)``.
        points_2d:
            ``(B, T, V, J, 2)`` 2D keypoints used to compute ray embeddings.
        view_mask:
            Optional binary mask of shape ``(B, T, V)`` or ``(B, V)``.  A value
            of ``0`` means the view is absent.

        Returns
        -------
        ``(B, T, V, J, d)`` updated tokens.  Masked-out views are zeroed.
        """
        B, T, V, J, d = x.shape
        device = x.device

        # Build ray/camera embedding if geometry is provided.
        if (
            K is not None
            and R is not None
            and t is not None
            and points_2d is not None
        ):
            ray_emb = self._build_ray_embedding(K, R, t, points_2d)
        else:
            ray_emb = torch.zeros(B, T, V, J, self.d, device=device, dtype=x.dtype)

        x_tokens = self._reshape_to_tokens(x)
        ray_tokens = self._reshape_to_tokens(ray_emb)

        key_mask: Optional[torch.Tensor] = None
        if view_mask is not None:
            key_mask = self._prepare_key_padding_mask(view_mask, B, T, V, J)

        for layer in self.layers:
            x_tokens = layer(x_tokens, ray_tokens, key_padding_mask=key_mask)

        # Zero out masked views so they cannot leak downstream.
        if key_mask is not None:
            x_tokens = x_tokens.masked_fill(key_mask.unsqueeze(-1), 0.0)

        return self._reshape_from_tokens(x_tokens, B, T, V, J, d)


if __name__ == "__main__":
    B, T, V, J, d = 2, 5, 4, 17, 64
    x = torch.randn(B, T, V, J, d)
    K = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(B * T, V, 3, 3)
    R = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(B * T, V, 3, 3)
    t = torch.randn(B * T, V, 3)
    points_2d = torch.randn(B, T, V, J, 2)

    module = CrossViewTransformerV17(d=d, n_heads=4, n_layers=2, dropout=0.0)
    out = module(x, K=K, R=R, t=t, points_2d=points_2d)
    assert out.shape == (B, T, V, J, d)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in module.parameters())
    print("CrossViewTransformerV17 CPU smoke test passed")
