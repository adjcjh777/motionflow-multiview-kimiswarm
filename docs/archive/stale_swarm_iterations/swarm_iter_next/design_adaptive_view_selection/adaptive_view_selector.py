"""Adaptive view selector prototype for MotionFlow-MultiView.

This module can be dropped into the combined cross-view + uncertainty +
residual + learned-triangulation model.  It predicts a per-view, per-joint
binary selection mask from spatio-temporal encoder tokens, optionally using
ray geometry.  During training a Gumbel-softmax relaxation is used; during
evaluation a hard top-k mask is returned.

Shape conventions follow the existing fusion models:
    feat:   (N, V, J, d)   encoder tokens (N = B*T)
    K,R,t:  (N, V, 3, 3), (N, V, 3, 3), (N, V, 3)
    points_2d: (N, V, J, 2)
    mask:   (N, V, J)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_ray_geometry(points_2d, K, R, t):
    """Return per-view, per-joint ray geometry features: ray angle, baseline, depth."""
    N, V, J, _ = points_2d.shape
    ones = torch.ones(N, V, J, 1, device=points_2d.device, dtype=points_2d.dtype)
    xy1 = torch.cat([points_2d, ones], dim=-1)  # (N, V, J, 3)

    K_inv = torch.inverse(K)
    d_cam = torch.einsum("nvic,nvkc->nvki", K_inv, xy1)
    d_world = torch.einsum("nvic,nvkc->nvki", R.transpose(-2, -1), d_cam)
    rays = d_world / (d_world.norm(dim=-1, keepdim=True) + 1e-8)

    centers = -torch.einsum("nvij,nvj->nvi", R.transpose(-2, -1), t)
    baselines = centers.unsqueeze(1) - centers.unsqueeze(2)  # (N, V, V, 3)
    baseline_len = baselines.norm(dim=-1).mean(dim=-1)  # (N, V)

    ray_dot = torch.einsum("nvjd,nujd->njv", rays, rays)  # average over other views
    ray_angle = torch.acos(torch.clamp(ray_dot, -1.0, 1.0))

    baseline_len = baseline_len.unsqueeze(-1).expand(N, V, J)
    ray_angle = ray_angle.permute(0, 2, 1)  # (N, V, J)
    return torch.stack([baseline_len, ray_angle], dim=-1)  # (N, V, J, 2)


class AdaptiveViewSelector(nn.Module):
    """Predict a per-view per-joint selection mask.

    Parameters
    ----------
    d: int
        Feature dimension from the encoder.
    n_views: int
        Maximum / expected number of views.
    k: int
        Number of views to select at inference time.
    tau: float
        Gumbel-softmax temperature.
    geo_features: bool
        Whether to concatenate ray-geometry features to the token.
    hard_inference: bool
        If True, use hard topk mask at inference; otherwise keep soft.
    """

    def __init__(
        self,
        d: int,
        n_views: int,
        k: int = 4,
        tau: float = 0.5,
        geo_features: bool = True,
        hard_inference: bool = True,
    ):
        super().__init__()
        self.d = d
        self.n_views = n_views
        self.k = k
        self.tau = tau
        self.geo_features = geo_features
        self.hard_inference = hard_inference

        geo_dim = 2 if geo_features else 0
        self.score_mlp = nn.Sequential(
            nn.Linear(d + geo_dim, d),
            nn.ReLU(),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, feat, points_2d=None, K=None, R=None, t=None):
        """Return (soft_mask, hard_mask, scores).

        feat: (N, V, J, d)
        points_2d, K, R, t: optional geometry inputs
        """
        N, V, J, d = feat.shape

        if self.geo_features:
            if points_2d is None or K is None or R is None or t is None:
                raise ValueError("geometry features requested but not provided")
            geo = _compute_ray_geometry(points_2d, K, R, t)
            x = torch.cat([feat, geo], dim=-1)
        else:
            x = feat

        # Per-view score for each joint.
        scores = self.score_mlp(x).squeeze(-1)  # (N, V, J)

        if self.training:
            # Gumbel-softmax over views; add a dummy "not selected" option to allow k=0.
            logits = torch.stack([scores, torch.zeros_like(scores)], dim=-1)  # (N, V, J, 2)
            soft = F.gumbel_softmax(logits, tau=self.tau, hard=False)  # (N, V, J, 2)
            soft_mask = soft[..., 0]  # "selected" channel
            # Straight-through hard mask preserves gradient path.
            hard_mask = (soft_mask > 0.5).float() - soft_mask.detach() + soft_mask
        else:
            if self.hard_inference:
                # Topk selection, straight-through gradient w.r.t. soft mask.
                _, topk_idx = torch.topk(scores, min(self.k, V), dim=1)  # (N, k, J)
                hard_mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)
                # Soft surrogate for gradient-free analysis.
                soft_mask = F.softmax(scores, dim=1)
            else:
                soft_mask = F.softmax(scores, dim=1)
                hard_mask = soft_mask

        return soft_mask, hard_mask, scores


if __name__ == "__main__":
    N, V, J, d = 2, 4, 17, 64
    feat = torch.randn(N, V, J, d)
    points_2d = torch.rand(N, V, J, 2) * 640
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(N, V, 3, 3)
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(N, V, 3, 3)
    t = torch.zeros(N, V, 3)

    selector = AdaptiveViewSelector(d=d, n_views=V, k=2)
    soft, hard, scores = selector(feat, points_2d, K, R, t)
    assert soft.shape == (N, V, J)
    assert hard.shape == (N, V, J)
    print("adaptive view selector sanity check passed")
    print("selected per joint:", hard.sum(dim=1)[0].tolist())
