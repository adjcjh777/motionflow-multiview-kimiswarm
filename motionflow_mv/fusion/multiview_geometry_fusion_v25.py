"""v25: Multi-View Geometry Fusion prototype.

A minimal but functional first implementation of the geometry-centric fusion
module described in ``docs/proposals/v25_multiview_geometry_fusion.md``.

The module keeps the *identity-at-init* property: when the learned weights are
zeroed, the block returns the input triangulated estimate unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .epipolar_attention_bias import compute_epipolar_distance
from .outlier_view_detector import OutlierViewDetector
from .triangulation import triangulate_dlt_batched_lstsq
from .uncertainty_depth_proposal_v27 import UncertaintyDepthProposalTriangulation


def _safe_inverse(x: torch.Tensor) -> torch.Tensor:
    """Batched matrix inverse with a small jitter for numerical stability."""
    return torch.inverse(x + 1e-7 * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype))


def build_projection_matrix(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build projection matrices P = K [R | t].

    Args:
        K: (B, T, V, 3, 3) intrinsics.
        R: (B, T, V, 3, 3) rotations.
        t: (B, T, V, 3) translations.

    Returns:
        P: (B, T, V, 3, 4) projection matrices.
    """
    RT = torch.cat([R, t[..., None]], dim=-1)  # (B, T, V, 3, 4)
    return torch.matmul(K, RT)


def triangulate_initial(
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triangulate 3D joints with batched DLT.

    Args:
        points_2d: (B, T, V, J, 2).
        K: (B, T, V, 3, 3).
        R: (B, T, V, 3, 3).
        t: (B, T, V, 3).
        weights: optional (B, T, V, J).

    Returns:
        X: (B, T, J, 3).
    """
    B, T, V, J, _ = points_2d.shape
    P = build_projection_matrix(K, R, t)  # (B, T, V, 3, 4)
    # triangulate_dlt_batched_lstsq expects (N, V, J, 2) and (N, V, 3, 4)
    P = P.reshape(B * T, V, 3, 4)
    pts = points_2d.reshape(B * T, V, J, 2)
    if weights is not None:
        weights = weights.reshape(B * T, V, J)
    X = triangulate_dlt_batched_lstsq(pts, P, weights=weights)
    # X: (B*T, J, 3)
    return X.reshape(B, T, J, 3)


def compute_rays(
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute camera centres and world ray directions for each view/joint.

    Args:
        points_2d: (B, T, V, J, 2).
        K: (B, T, V, 3, 3).
        R: (B, T, V, 3, 3).
        t: (B, T, V, 3).

    Returns:
        centre: (B, T, V, 3) camera centre c = -R^T t.
        direction: (B, T, V, J, 3) unit world ray direction.
    """
    # c = -R^T t
    centre = -torch.matmul(R.transpose(-2, -1), t[..., None]).squeeze(-1)  # (B, T, V, 3)

    # Homogeneous image point and inverse intrinsics.
    ones = torch.ones(*points_2d.shape[:-1], 1, device=points_2d.device, dtype=points_2d.dtype)
    pts_h = torch.cat([points_2d, ones], dim=-1)  # (B, T, V, J, 3)
    K_inv = _safe_inverse(K)  # (B, T, V, 3, 3)
    # d = R^T K^{-1} p
    direction = torch.matmul(K_inv[:, :, :, None, :], pts_h[..., None])  # (B, T, V, J, 3, 1)
    direction = direction.squeeze(-1)
    direction = torch.matmul(R.transpose(-2, -1)[:, :, :, None, :], direction[..., None]).squeeze(-1)
    direction = F.normalize(direction, dim=-1)
    return centre, direction


def ray_intersection_logit(
    centre: torch.Tensor,
    direction: torch.Tensor,
    sigma_d: torch.Tensor,
    sigma_a: torch.Tensor,
) -> torch.Tensor:
    """Compute ray-intersection quality between all view pairs.

    Args:
        centre: (B, T, V, 3).
        direction: (B, T, V, J, 3).
        sigma_d: () learnable distance temperature.
        sigma_a: () learnable angle temperature.

    Returns:
        logit: (B, T, V, V, J) additive attention logit. Higher means rays
            are closer / more compatible.
    """
    # Baseline vector between camera centres.
    c_i = centre[:, :, :, None, :]  # (B, T, V_i, 1, 3)
    c_j = centre[:, :, None, :, :]  # (B, T, 1, V_j, 3)
    d_i = direction[:, :, :, None, :]  # (B, T, V_i, 1, J, 3)
    d_j = direction[:, :, None, :, :]  # (B, T, 1, V_j, J, 3)

    # Shortest distance between two skew rays.
    # See e.g. Hartley & Zisserman.
    w = c_j - c_i  # (B, T, V, V, 3)
    cross = torch.cross(d_i, d_j, dim=-1)  # (B, T, V, V, J, 3)
    denom = torch.linalg.norm(cross, dim=-1).clamp(min=1e-6)  # (B, T, V, V, J)
    dist = torch.abs((w[:, :, :, :, None, :] * cross).sum(dim=-1)) / denom  # (B, T, V, V, J)

    cos = (d_i * d_j).sum(dim=-1).clamp(min=-1.0, max=1.0)  # (B, T, V, V, J)
    logit = -(dist / sigma_d) - ((1.0 - cos) / sigma_a)
    return logit  # (B, T, V, V, J)


class RayTokenizer(nn.Module):
    """Encode per-view rays into d-dimensional tokens.

    The depth embedding is represented by a small bank of learnable depth
    hypotheses that are projected along the ray and collapsed by a 1x1 conv.
    """

    def __init__(self, d: int = 128, n_ray_samples: int = 4):
        super().__init__()
        self.n_ray_samples = n_ray_samples
        # Per-sample depth codes; a tiny 1D conv merges them.
        self.depth_codes = nn.Parameter(torch.randn(n_ray_samples, max(1, d // 4)))
        self.depth_mlp = nn.Sequential(
            nn.Linear(max(1, d // 4), d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        # Final ray token combines direction, camera centre, confidence and depth.
        self.token_mlp = nn.Sequential(
            nn.Linear(3 + 3 + 1 + d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(
        self,
        centre: torch.Tensor,
        direction: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            centre: (B, T, V, 3).
            direction: (B, T, V, J, 3).
            confidence: (B, T, V, J).

        Returns:
            tokens: (B, T, V, J, d).
        """
        B, T, V, J = direction.shape[:4]
        # depth codes -> (n_ray_samples, d) -> aggregate.
        depth_emb = self.depth_mlp(self.depth_codes).mean(dim=0)  # (d,)
        depth_emb = depth_emb.view(1, 1, 1, 1, 1, -1).expand(B, T, V, J, 1, -1)

        centre = centre[:, :, :, None, :].expand(-1, -1, -1, J, -1)
        conf = confidence[..., None]
        base = torch.cat([direction, centre, conf], dim=-1)  # (B, T, V, J, 7)
        base = base.unsqueeze(4).expand(-1, -1, -1, -1, 1, -1)
        x = torch.cat([base, depth_emb], dim=-1)  # (B, T, V, J, 1, 7+d)
        tokens = self.token_mlp(x.squeeze(4))
        return tokens


class GeometryAwareCrossViewAttention(nn.Module):
    """Self-attention over views biased by epipolar and ray-intersection logits."""

    def __init__(self, d: int, n_heads: int, n_views: int, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d // n_heads)

        # Learnable geometry temperatures.
        self.sigma_d = nn.Parameter(torch.tensor(0.5))
        self.sigma_a = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)

        # Identity at init: zero the output projection so attention residual vanishes.
        for p in self.out_proj.parameters():
            nn.init.zeros_(p)

        # Optional debug hook for visualising / inspecting the attention map.
        self.record_attention = False
        self.last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        tokens: torch.Tensor,
        epipolar_dist: torch.Tensor,
        ray_logit: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tokens: (B, T, V, J, d).
            epipolar_dist: (B, T, V, V, J) epipolar-line distances.
            ray_logit: (B, T, V, V, J) ray-intersection logit.
            view_mask: optional (B, T, V) bool; True = keep.

        Returns:
            refined: (B, T, V, J, d).
        """
        B, T, V, J, d = tokens.shape
        # Per-joint linear projections.
        qkv = self.qkv(tokens)  # (B, T, V, J, 3d)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, V, J, d)
        # Reshape for multi-head: (B*T*J, V, d_h)
        q = q.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * T * J, V, d)
        q = q.view(B * T * J, V, self.n_heads, d // self.n_heads).transpose(1, 2)
        k = k.view(B * T * J, V, self.n_heads, d // self.n_heads).transpose(1, 2)
        v = v.view(B * T * J, V, self.n_heads, d // self.n_heads).transpose(1, 2)
        # content scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (N, h, V, V)

        # geometry bias per joint
        # epipolar: lower distance -> higher bias
        epi_bias = -(epipolar_dist.permute(0, 1, 4, 2, 3).reshape(B * T * J, V, V))  # (N, V, V)
        ray_bias = ray_logit.permute(0, 1, 4, 2, 3).reshape(B * T * J, V, V)  # (N, V, V)
        scores = scores + epi_bias.unsqueeze(1) + ray_bias.unsqueeze(1)

        if view_mask is not None:
            # mask: (B, T, V) -> (B*T*J, V)
            mask = view_mask.reshape(B * T, V)
            mask = mask[:, None, None, :].expand(-1, 1, J, -1).reshape(B * T * J, 1, 1, V)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.record_attention:
            self.last_attention = attn.detach().cpu()
        out = torch.matmul(attn, v)  # (N, h, V, d_h)
        out = out.transpose(1, 2).reshape(B * T * J, V, d)
        out = self.out_proj(out)
        out = out.view(B, T, J, V, d).permute(0, 1, 3, 2, 4)  # (B, T, V, J, d)
        return self.dropout(out)


class DepthProposalTriangulation(nn.Module):
    """Learned depth-proposal triangulation head.

    Samples a small set of depth hypotheses along each ray, scores them with a
    tiny MLP, and aggregates the candidates. The final layer is initialised to
    zero so the head starts as an identity/no-op w.r.t. the input 3D estimate.
    """

    def __init__(self, n_views: int, n_ray_samples: int = 4):
        super().__init__()
        self.n_views = n_views
        self.n_ray_samples = n_ray_samples
        # Learnable depth sample grid; kept close to a reasonable metre range.
        self.z_min = nn.Parameter(torch.tensor(1.0))
        self.z_max = nn.Parameter(torch.tensor(8.0))
        # Score each (view, sample) candidate from candidate + per-view context.
        in_dim = 3 + 3 + 1 + 3
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Zero final layer -> scores all equal at init.
        for p in self.score_mlp[-1].parameters():
            nn.init.zeros_(p)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        # Scalar gate initialised to 0.0 gives identity at init; training opens it.
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        centre: torch.Tensor,
        direction: torch.Tensor,
        confidence: torch.Tensor,
        pred_3d: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            centre: (B, T, V, 3).
            direction: (B, T, V, J, 3).
            confidence: (B, T, V, J).
            pred_3d: (B, T, J, 3).
            view_mask: optional (B, T, V).

        Returns:
            refined: (B, T, J, 3).
        """
        B, T, V, J = direction.shape[:4]
        # Build depth samples and candidate points along each ray.
        z_vals = torch.linspace(0.0, 1.0, self.n_ray_samples, device=direction.device, dtype=direction.dtype)
        z_vals = self.z_min + (self.z_max - self.z_min) * z_vals  # (S,)
        # c + z d; shapes (B, T, V, J, S, 3)
        candidates = (
            centre[:, :, :, None, None, :]
            + z_vals.view(1, 1, 1, 1, -1, 1) * direction[:, :, :, :, None, :]
        )  # (B, T, V, J, S, 3)

        # Aggregate per-view context: mean candidate and deviation from current estimate.
        mean_candidate = candidates.mean(dim=4, keepdim=True).expand(-1, -1, -1, -1, self.n_ray_samples, -1)
        pred_exp = pred_3d[:, :, None, :, None, :].expand(-1, -1, V, -1, self.n_ray_samples, -1)
        conf_exp = confidence[..., None, None].expand(-1, -1, -1, -1, self.n_ray_samples, 1)
        feat = torch.cat([candidates, mean_candidate, pred_exp, conf_exp], dim=-1)
        scores = self.score_mlp(feat).squeeze(-1)  # (B, T, V, J, S)

        if view_mask is not None:
            scores = scores.masked_fill(~view_mask[:, :, :, None, None], float("-inf"))

        # Softmax over the (V, S) candidate dimension, not over joints.
        scores_flat = scores.view(B, T, V * self.n_ray_samples, J).permute(0, 1, 3, 2)  # (B, T, J, V*S)
        # Guard against rows that are all -inf (fully masked); leave them as uniform zero weight.
        scores_flat = torch.where(
            torch.isinf(scores_flat).all(dim=-1, keepdim=True),
            torch.zeros_like(scores_flat),
            scores_flat,
        )
        probs = F.softmax(scores_flat, dim=-1)
        # Weighted average over (V, S).
        candidates_flat = candidates.view(B, T, V * self.n_ray_samples, J, 3).permute(0, 1, 3, 2, 4)
        fused = (probs[..., None] * candidates_flat).sum(dim=3)  # (B, T, J, 3)

        # Identity-at-init residual around the input estimate.
        residual = self.fusion_mlp(fused - pred_3d)
        return pred_3d + self.residual_scale * residual


class MultiViewGeometryFusionV25(nn.Module):
    """Geometry-centric multi-view fusion module (v25 prototype).

    Parameters
    ----------
    d: feature dimension used by ray tokens.
    n_heads: number of attention heads in geometry-aware cross-view attention.
    n_views: number of views (used for shape hints).
    n_geometry_layers: number of stacked geometry-attention layers.
    n_ray_samples: depth hypotheses per ray in the learned triangulation head.
    use_geometry_attention: enable geometry-aware cross-view attention.
    use_learned_depth_triangulation: enable learned depth-proposal triangulation.
    use_geometry_bundle_adjustment: reserved placeholder (currently no-op).
    use_outlier_view_detector: enable robust outlier-view detection and down-weighting.
    outlier_z_thresh: robust z-score threshold for the outlier detector.
    outlier_soft_beta: softness of the exponential outlier down-weighting.
    dropout: dropout rate.
    """

    def __init__(
        self,
        d: int = 128,
        n_heads: int = 4,
        n_views: int = 4,
        n_geometry_layers: int = 2,
        n_ray_samples: int = 4,
        use_geometry_attention: bool = True,
        use_learned_depth_triangulation: bool = True,
        use_geometry_bundle_adjustment: bool = False,
        use_camera_joint_graph: bool = False,
        use_outlier_view_detector: bool = False,
        outlier_z_thresh: float = 3.0,
        outlier_soft_beta: float = 1.0,
        dropout: float = 0.1,
        use_uncertainty_depth_proposals_v27: bool = False,
        v27_uncertainty_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.n_views = n_views
        self.use_geometry_attention = use_geometry_attention
        self.use_learned_depth_triangulation = use_learned_depth_triangulation
        self.use_geometry_bundle_adjustment = use_geometry_bundle_adjustment
        self.use_camera_joint_graph = use_camera_joint_graph
        self.use_outlier_view_detector = use_outlier_view_detector
        self.use_uncertainty_depth_proposals_v27 = use_uncertainty_depth_proposals_v27
        self.v27_uncertainty_loss_weight = v27_uncertainty_loss_weight

        self.ray_tokenizer = RayTokenizer(d=d, n_ray_samples=n_ray_samples)

        if use_geometry_attention:
            self.geom_attn_layers = nn.ModuleList(
                [GeometryAwareCrossViewAttention(d, n_heads, n_views, dropout) for _ in range(n_geometry_layers)]
            )
        else:
            self.geom_attn_layers = None

        if use_learned_depth_triangulation:
            if use_uncertainty_depth_proposals_v27:
                self.depth_tri_head = UncertaintyDepthProposalTriangulation(
                    n_views=n_views, n_ray_samples=n_ray_samples, uncertainty_loss_weight=v27_uncertainty_loss_weight
                )
            else:
                self.depth_tri_head = DepthProposalTriangulation(n_views=n_views, n_ray_samples=n_ray_samples)
        else:
            self.depth_tri_head = None

        if use_outlier_view_detector:
            self.outlier_view_detector = OutlierViewDetector(z_thresh=outlier_z_thresh, soft_beta=outlier_soft_beta)
        else:
            self.outlier_view_detector = None

    def forward(
        self,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        pred_3d_init: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the v25 geometry fusion block.

        Args:
            points_2d: (B, T, V, J, 2) or (B, T, V, J, 3). If the last dim is 3,
                the third channel is interpreted as per-joint confidence.
            K: (B, T, V, 3, 3) intrinsics.
            R: (B, T, V, 3, 3) rotations.
            t: (B, T, V, 3) translations.
            pred_3d_init: optional (B, T, J, 3) initial triangulated estimate. If
                not provided, it is computed via DLT from ``points_2d``.
            view_mask: optional (B, T, V) bool mask. True / 1 = view is valid.
            confidence: optional (B, T, V, J) confidence weights.

        Returns:
            pred_3d_ref: (B, T, J, 3) refined 3D joints.
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
            # Convert to bool if needed.
            view_mask = view_mask.bool()

        # Initial triangulation if not supplied.
        if pred_3d_init is None:
            tri_weights = confidence if view_mask is None else confidence * view_mask[:, :, :, None]
            pred_3d_init = triangulate_initial(pts, K, R, t, weights=tri_weights)

        # Optional outlier-view detection and down-weighting.
        if self.use_outlier_view_detector and self.outlier_view_detector is not None:
            outlier_weights, _ = self.outlier_view_detector(pred_3d_init, pts, K, R, t, view_mask=view_mask)
            confidence = confidence * outlier_weights
            # Re-triangulate with outlier-down-weighted confidence for a cleaner seed.
            tri_weights = confidence if view_mask is None else confidence * view_mask[:, :, :, None]
            pred_3d_init = triangulate_initial(pts, K, R, t, weights=tri_weights)

        # World rays.
        centre, direction = compute_rays(pts, K, R, t)

        # Ray tokens.
        tokens = self.ray_tokenizer(centre, direction, confidence)

        # Geometry-aware cross-view attention on ray tokens.
        if self.use_geometry_attention and self.geom_attn_layers is not None:
            epipolar_dist = compute_epipolar_distance(
                K.reshape(B * T, V, 3, 3),
                R.reshape(B * T, V, 3, 3),
                t.reshape(B * T, V, 3),
                pts.reshape(B * T, V, J, 2),
            )  # (B*T, V, V, J)
            epipolar_dist = epipolar_dist.reshape(B, T, V, V, J)
            ray_logit = ray_intersection_logit(centre, direction, self.geom_attn_layers[0].sigma_d, self.geom_attn_layers[0].sigma_a)
            for layer in self.geom_attn_layers:
                tokens = tokens + layer(tokens, epipolar_dist, ray_logit, view_mask=view_mask)

        # Learned depth-proposal triangulation refines the initial 3D estimate.
        uncertainty_loss = torch.tensor(0.0, device=pts.device, dtype=pts.dtype)
        if self.use_learned_depth_triangulation and self.depth_tri_head is not None:
            if self.use_uncertainty_depth_proposals_v27:
                pred_3d_ref, uncertainty_loss = self.depth_tri_head(
                    centre, direction, confidence, pred_3d_init, view_mask=view_mask
                )
            else:
                pred_3d_ref = self.depth_tri_head(centre, direction, confidence, pred_3d_init, view_mask=view_mask)
        else:
            pred_3d_ref = pred_3d_init

        geom_loss = self._reprojection_loss(pred_3d_ref, pts, K, R, t, confidence, view_mask)
        return pred_3d_ref, geom_loss + uncertainty_loss

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
        # X_cam = R @ X + t, then project with K after dividing by Z.
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
        # Normalise by a canonical image scale so the loss is O(1).
        loss = loss / 1000.0
        return loss
