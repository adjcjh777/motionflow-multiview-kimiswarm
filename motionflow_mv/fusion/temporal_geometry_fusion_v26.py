"""v26: Temporal Multi-View Geometry Fusion prototype.

Extends ``MultiViewGeometryFusionV25`` by adding a lightweight
spatio-temporal geometry attention block. Each (time, view, joint) query
attends to a small temporal window of neighbouring frames across all views.
The attention is biased by epipolar distance, ray-intersection quality and a
learned temporal offset, so geometry drives the sparse temporal fusion.

The module keeps the *identity-at-init* property of v25: with zeroed learned
weights it reduces to the v25 per-frame geometry fusion.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .epipolar_attention_bias import compute_epipolar_distance
from .multiview_geometry_fusion_v25 import (
    DepthProposalTriangulation,
    GeometryAwareCrossViewAttention,
    RayTokenizer,
    compute_rays,
    ray_intersection_logit,
    triangulate_initial,
)


class TemporalGeometryAttention(nn.Module):
    """Spatio-temporal geometry attention over (time, view) pairs per joint.

    For each query token at frame ``t`` and view ``v_q``, the module gathers
    key tokens from frames ``t - half_window, ..., t + half_window`` across all
    views. Content scores are biased by per-frame epipolar distance and
    ray-intersection logits plus a learnable temporal-offset bias.

    Parameters
    ----------
    d:
        Token dimension.
    n_heads:
        Number of attention heads.
    n_views:
        Number of views.
    temporal_window:
        Size of the temporal window; must be odd. Default ``3`` corresponds to
        offsets ``[-1, 0, +1]``.
    dropout:
        Dropout probability on the output projection.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_views: int,
        temporal_window: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        if temporal_window % 2 == 0:
            raise ValueError("temporal_window must be odd")

        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.temporal_window = temporal_window
        self.half = temporal_window // 2

        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d // n_heads)

        # Learnable bias for each temporal offset. Initialised near zero so the
        # module starts as a near-identity residual.
        self.temporal_pos = nn.Parameter(torch.zeros(temporal_window))

        self.dropout = nn.Dropout(dropout)

        # Identity at init: zero the output projection so the residual vanishes.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _build_temporal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Return (T, temporal_window) bool mask for in-bound temporal offsets."""
        t_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        offsets = torch.arange(-self.half, self.half + 1, device=device).unsqueeze(0)  # (1, W)
        valid = (t_idx + offsets >= 0) & (t_idx + offsets < T)  # (T, W)
        return valid

    def forward(
        self,
        tokens: torch.Tensor,
        epipolar_dist: torch.Tensor,
        ray_logit: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run spatio-temporal geometry attention.

        Args:
            tokens: (B, T, V, J, d).
            epipolar_dist: (B, T, V, V, J) epipolar-line distances.
            ray_logit: (B, T, V, V, J) ray-intersection logit.
            view_mask: optional (B, T, V) bool; True = keep.

        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        W = self.temporal_window
        half = self.half
        N = B * J
        dh = d // self.n_heads

        # Compute Q, K, V and arrange as (B*J, T, V, d).
        qkv = self.qkv(tokens).chunk(3, dim=-1)
        q, k, v = [t.permute(0, 3, 1, 2, 4).reshape(N, T, V, d) for t in qkv]

        # Pad temporally with zeros for boundary frames.
        k_padded = F.pad(k, (0, 0, 0, 0, half, half))  # (N, T+W-1, V, d)
        v_padded = F.pad(v, (0, 0, 0, 0, half, half))

        # Gather windows: for query frame t the window is original frames
        # [t-half, ..., t+half]. With zero padding, out-of-bound frames are zero.
        k_win = torch.stack([k_padded[:, w : w + T, :, :] for w in range(W)], dim=2)
        v_win = torch.stack([v_padded[:, w : w + T, :, :] for w in range(W)], dim=2)
        # (N, T, W, V, d) -> (N, T, W*V, d)
        k_win = k_win.reshape(N, T, W * V, d)
        v_win = v_win.reshape(N, T, W * V, d)

        # Arrange query: (N, h, T, V, dh)
        q = q.reshape(N, T, V, self.n_heads, dh).permute(0, 3, 1, 2, 4)
        # Arrange keys/values: (N, h, T, W*V, dh)
        k_win = k_win.reshape(N, T, W * V, self.n_heads, dh).permute(0, 3, 1, 2, 4)
        v_win = v_win.reshape(N, T, W * V, self.n_heads, dh).permute(0, 3, 1, 2, 4)

        # Content scores: for each frame, q (V, dh) @ k (W*V, dh)^T
        # -> (N, h, T, V, W*V)
        scores = torch.matmul(q, k_win.transpose(-2, -1)) / self.scale

        # Geometry bias: per query view v_q and key view v_k, replicated over the
        # temporal window. Shapes: (B, J, T, V, V) -> (N, T, V, V).
        epi_bias = -epipolar_dist.permute(0, 4, 1, 2, 3).reshape(N, T, V, V)
        ray_bias = ray_logit.permute(0, 4, 1, 2, 3).reshape(N, T, V, V)
        # Expand over temporal window and reshape to (N, T, V, W*V).
        epi_bias = epi_bias.unsqueeze(2).expand(-1, -1, W, -1, -1)
        epi_bias = epi_bias.permute(0, 1, 3, 2, 4).reshape(N, T, V, W * V)
        ray_bias = ray_bias.unsqueeze(2).expand(-1, -1, W, -1, -1)
        ray_bias = ray_bias.permute(0, 1, 3, 2, 4).reshape(N, T, V, W * V)
        scores = scores + epi_bias.unsqueeze(1) + ray_bias.unsqueeze(1)

        # Temporal offset bias.
        temp_bias = self.temporal_pos[:, None].expand(-1, V).reshape(W * V)  # (W*V,)
        scores = scores + temp_bias

        # Build the attention mask over keys (W*V) and broadcast to scores.
        # temporal_valid: (T, W) -> (1, 1, T, 1, W*V)
        temporal_valid = self._build_temporal_mask(T, tokens.device)
        temporal_valid = temporal_valid[:, :, None].expand(-1, -1, V).reshape(T, W * V)
        mask = temporal_valid[None, None, :, None, :].expand(N, 1, T, V, W * V)

        if view_mask is not None:
            # view_mask: (B, T, V) -> (N, T, V) -> key mask (N, 1, T, W*V)
            vm = view_mask[:, None, :, :].expand(-1, J, -1, -1).reshape(N, T, V)
            vm = vm[:, None, :, None, :].expand(-1, self.n_heads, -1, W, -1)
            vm = vm.reshape(N, self.n_heads, T, 1, W * V)
            mask = mask & vm

        scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax and aggregate.
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v_win)  # (N, h, T, V, dh)

        # Reshape back to (B, T, V, J, d).
        out = out.permute(0, 2, 3, 1, 4).reshape(N, T, V, d)
        out = self.out_proj(out)
        out = out.view(B, J, T, V, d).permute(0, 2, 3, 1, 4)
        return self.dropout(out)


class TemporalGeometryFusionV26(nn.Module):
    """Temporal extension of v25 geometry fusion (v26 prototype).

    Parameters
    ----------
    d:
        Feature dimension.
    n_heads:
        Number of attention heads.
    n_views:
        Number of views.
    n_geometry_layers:
        Number of per-frame geometry-attention layers (v25 component).
    n_temporal_layers:
        Number of spatio-temporal attention layers.
    n_ray_samples:
        Depth hypotheses per ray in the learned triangulation head.
    temporal_window:
        Temporal window size for spatio-temporal attention.
    use_geometry_attention:
        Enable v25 per-frame geometry-aware cross-view attention.
    use_temporal_geometry_attention:
        Enable spatio-temporal geometry attention.
    use_learned_depth_triangulation:
        Enable v25 learned depth-proposal triangulation.
    use_temporal_depth_consistency:
        If True, add a temporal velocity-smoothness loss on the refined 3D
        trajectory.
    temporal_loss_weight:
        Weight of the temporal velocity-smoothness loss.
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        n_geometry_layers: int = 2,
        n_temporal_layers: int = 1,
        n_ray_samples: int = 4,
        temporal_window: int = 3,
        use_geometry_attention: bool = True,
        use_temporal_geometry_attention: bool = True,
        use_learned_depth_triangulation: bool = True,
        use_temporal_depth_consistency: bool = False,
        temporal_loss_weight: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.use_geometry_attention = use_geometry_attention
        self.use_temporal_geometry_attention = use_temporal_geometry_attention
        self.use_learned_depth_triangulation = use_learned_depth_triangulation
        self.use_temporal_depth_consistency = use_temporal_depth_consistency
        self.temporal_loss_weight = temporal_loss_weight

        self.ray_tokenizer = RayTokenizer(d=d, n_ray_samples=n_ray_samples)

        if use_geometry_attention:
            self.geom_attn_layers = nn.ModuleList(
                [GeometryAwareCrossViewAttention(d, n_heads, n_views, dropout) for _ in range(n_geometry_layers)]
            )
        else:
            self.geom_attn_layers = None

        if use_temporal_geometry_attention:
            self.temporal_attn_layers = nn.ModuleList(
                [
                    TemporalGeometryAttention(d, n_heads, n_views, temporal_window, dropout)
                    for _ in range(n_temporal_layers)
                ]
            )
        else:
            self.temporal_attn_layers = None

        if use_learned_depth_triangulation:
            self.depth_tri_head = DepthProposalTriangulation(n_views=n_views, n_ray_samples=n_ray_samples)
        else:
            self.depth_tri_head = None

    def forward(
        self,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        pred_3d_init: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the v26 temporal geometry fusion block.

        Args:
            points_2d: (B, T, V, J, 2) or (B, T, V, J, 3). If the last dim is 3,
                the third channel is interpreted as per-joint confidence.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            pred_3d_init: optional (B, T, J, 3) initial triangulated estimate.
            view_mask: optional (B, T, V) bool mask. True / 1 = view is valid.
            confidence: optional (B, T, V, J) confidence weights.

        Returns:
            pred_3d_ref: (B, T, J, 3) refined 3D joints.
            geom_loss: scalar geometry-aware loss (reprojection + temporal smoothness).
        """
        if points_2d.shape[-1] == 3:
            confidence = points_2d[..., 2]
            pts = points_2d[..., :2]
        else:
            pts = points_2d
        B, T, V, J, _ = pts.shape

        if confidence is None:
            confidence = torch.ones(B, T, V, J, device=pts.device, dtype=pts.dtype)

        if view_mask is not None:
            view_mask = view_mask.bool()

        if pred_3d_init is None:
            tri_weights = confidence if view_mask is None else confidence * view_mask[:, :, :, None]
            pred_3d_init = triangulate_initial(pts, K, R, t, weights=tri_weights)

        # World rays.
        centre, direction = compute_rays(pts, K, R, t)

        # Ray tokens.
        tokens = self.ray_tokenizer(centre, direction, confidence)

        # Per-frame geometry-aware cross-view attention (v25).
        epipolar_dist = None
        ray_logit = None
        if self.use_geometry_attention and self.geom_attn_layers is not None:
            epipolar_dist = compute_epipolar_distance(
                K.reshape(B * T, V, 3, 3),
                R.reshape(B * T, V, 3, 3),
                t.reshape(B * T, V, 3),
                pts.reshape(B * T, V, J, 2),
            )  # (B*T, V, V, J)
            epipolar_dist = epipolar_dist.reshape(B, T, V, V, J)
            sigma_d = self.geom_attn_layers[0].sigma_d
            sigma_a = self.geom_attn_layers[0].sigma_a
            ray_logit = ray_intersection_logit(centre, direction, sigma_d, sigma_a)
            for layer in self.geom_attn_layers:
                tokens = tokens + layer(tokens, epipolar_dist, ray_logit, view_mask=view_mask)

        # Spatio-temporal geometry attention (v26).
        if self.use_temporal_geometry_attention and self.temporal_attn_layers is not None:
            if epipolar_dist is None or ray_logit is None:
                epipolar_dist = compute_epipolar_distance(
                    K.reshape(B * T, V, 3, 3),
                    R.reshape(B * T, V, 3, 3),
                    t.reshape(B * T, V, 3),
                    pts.reshape(B * T, V, J, 2),
                ).reshape(B, T, V, V, J)
                ray_logit = ray_intersection_logit(
                    centre,
                    direction,
                    torch.tensor(0.5, device=centre.device),
                    torch.tensor(0.5, device=centre.device),
                )
            for layer in self.temporal_attn_layers:
                tokens = tokens + layer(tokens, epipolar_dist, ray_logit, view_mask=view_mask)

        # Learned depth-proposal triangulation refines the initial 3D estimate.
        if self.use_learned_depth_triangulation and self.depth_tri_head is not None:
            pred_3d_ref = self.depth_tri_head(centre, direction, confidence, pred_3d_init, view_mask=view_mask)
        else:
            pred_3d_ref = pred_3d_init

        geom_loss = self._reprojection_loss(pred_3d_ref, pts, K, R, t, confidence, view_mask)

        # Optional temporal smoothness loss on the refined 3D trajectory.
        if self.use_temporal_depth_consistency and T > 1:
            velocity = pred_3d_ref[:, 1:] - pred_3d_ref[:, :-1]
            temporal_loss = velocity.norm(dim=-1).mean()
            geom_loss = geom_loss + self.temporal_loss_weight * temporal_loss

        return pred_3d_ref, geom_loss

    def _reprojection_loss(
        self,
        X: torch.Tensor,
        pts_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        confidence: torch.Tensor,
        view_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean reprojection error of 3D joints weighted by confidence."""
        B, T, V, J = pts_2d.shape[:4]
        X_expanded = X.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B, T, V, J, 3)
        X_expanded = X_expanded.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
        X_cam = torch.matmul(R, X_expanded)  # (B, T, V, 3, J)
        X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
        X_cam = X_cam + t[..., None, :]
        Z = X_cam[..., 2:3]
        Z_safe = Z.sign() * (Z.abs() + 1e-6)
        X_norm = X_cam / Z_safe
        proj = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)  # (B, T, V, J, 3)
        proj_2d = proj[..., :2] / proj[..., 2:3]
        diff = (proj_2d - pts_2d).norm(dim=-1)  # (B, T, V, J)
        if view_mask is not None:
            mask = view_mask[:, :, :, None].float()
        else:
            mask = torch.ones(B, T, V, 1, device=X.device, dtype=X.dtype)
        weights = confidence * mask  # (B, T, V, J)
        loss = (diff * weights).sum() / weights.sum().clamp(min=1e-6)
        return loss / 1000.0
