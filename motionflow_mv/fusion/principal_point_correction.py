"""Principal-point correction layer for calibrated multi-view pose.

The pinhole calibration matrix ``K`` is usually assumed known and fixed.
Empirically, small errors in the principal point (cx, cy) are common
(off-the-shelf calibration, reprojection of pre-calibrated rigs, etc.) and
hurt triangulation accuracy.  This module learns a *small* per-view correction
``(dx, dy)`` that is added to the principal point before triangulation.

The correction can be predicted either from deep per-view features (e.g. the
output of a temporal transformer) or directly from per-view 2D observation
statistics.  The predicted offset is bounded with ``tanh`` so the layer stays
near the identity at initialization and never catastrophically redefines the
camera model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PrincipalPointCorrection(nn.Module):
    """Learn a bounded per-view principal-point offset ``(dx, dy)``.

    Parameters
    ----------
    d:
        Feature dimension when ``feat`` is provided.
    hidden:
        Hidden size of the offset predictor.
    max_offset:
        Maximum absolute correction in pixels.  The predictor output is
        ``tanh``-squashed to ``[-max_offset, +max_offset]``.
    use_confidence:
        If ``True``, pool ``feat`` using the supplied confidence/weight map;
        otherwise use uniform averaging over joints.
    """

    def __init__(
        self,
        d: int = 64,
        hidden: int = 64,
        max_offset: float = 20.0,
        max_focal_scale: float = 0.0,
        use_confidence: bool = True,
    ):
        super().__init__()
        self.d = d
        self.max_offset = max_offset
        self.max_focal_scale = max_focal_scale
        self.use_confidence = use_confidence

        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

        # Dedicated focal-length correction head, sharing the same pooled features.
        if max_focal_scale > 0:
            self.focal_mlp = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )
        else:
            self.focal_mlp = None

        # Fallback projector for the raw-observation path (8-D descriptor -> d).
        self.fallback_projector = nn.Linear(8, d)

    def forward(
        self,
        K: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict principal-point offsets and apply them to ``K``.

        Parameters
        ----------
        K:
            Intrinsic matrices ``(N, V, 3, 3)``.
        feat:
            Optional per-view per-joint features ``(N, V, J, d)``.
        x:
            Optional raw input ``(N, V, J, 3)`` with ``[..., :2]`` the 2D
            keypoints and ``[..., 2]`` the confidence.  Used only when
            ``feat`` is not provided.
        weights:
            Optional pooling weights ``(N, V, J)``.  If ``None`` and
            ``feat`` is provided, uniform averaging over joints is used.

        Returns
        -------
        K_corrected:
            Intrinsic matrices with updated principal points, same shape as K.
        delta:
            Predicted offsets ``(N, V, 2)`` in pixels.
        """
        if feat is not None:
            pooled = self._pool_features(feat, weights)
        elif x is not None:
            pooled = self._features_from_x(x, K, weights)
        else:
            raise ValueError("Either feat or x must be provided.")

        # import sys, traceback
        # print(f"[DEBUG pp] caller:\n{''.join(traceback.format_stack(limit=5)[:-1])}", file=sys.stderr, flush=True)
        # print(f"[DEBUG pp] K finite={torch.isfinite(K).all().item()} x finite={torch.isfinite(x).all().item() if x is not None else 'n/a'} weights finite={torch.isfinite(weights).all().item() if weights is not None else 'n/a'} pooled finite={torch.isfinite(pooled).all().item()} pooled shape={pooled.shape} min={pooled.min().item() if torch.isfinite(pooled).all() else 'nan'} max={pooled.max().item() if torch.isfinite(pooled).all() else 'nan'}", file=sys.stderr, flush=True)
        # if x is not None:
        #     print(f"[DEBUG pp] x sum={x.sum().item()} abs_sum={x.abs().sum().item()}", file=sys.stderr, flush=True)
        # if weights is not None:
        #     print(f"[DEBUG pp] weights sum={weights.sum().item()} abs_sum={weights.abs().sum().item()} any_neg={ (weights < 0).any().item()}", file=sys.stderr, flush=True)
        out = self.mlp(pooled)
        # print(f"[DEBUG pp] mlp_out finite={torch.isfinite(out).all().item()} out min={out.min().item() if torch.isfinite(out).all() else 'nan'} max={out.max().item() if torch.isfinite(out).all() else 'nan'}", file=sys.stderr, flush=True)
        delta = out[..., :2] * self.max_offset  # (N, V, 2)

        # Build the corrected intrinsics in a functional way to avoid in-place
        # operations that break the autograd graph when focal length is corrected.
        cx = K[..., 0, 2] + delta[..., 0]
        cy = K[..., 1, 2] + delta[..., 1]
        if self.max_focal_scale > 0:
            focal_scale = 1.0 + self.focal_mlp(pooled).squeeze(-1) * self.max_focal_scale  # (N, V)
            fx = K[..., 0, 0] * focal_scale
            fy = K[..., 1, 1] * focal_scale
        else:
            focal_scale = None
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]

        K_corrected = torch.stack(
            [
                torch.stack([fx, K[..., 0, 1], cx], dim=-1),
                torch.stack([K[..., 1, 0], fy, cy], dim=-1),
                torch.stack([K[..., 2, 0], K[..., 2, 1], K[..., 2, 2]], dim=-1),
            ],
            dim=-2,
        )

        if self.max_focal_scale > 0:
            return K_corrected, delta, focal_scale
        return K_corrected, delta

    def _pool_features(
        self,
        feat: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Weighted average over joints -> (N, V, d)."""
        if weights is not None:
            # Normalize per view.
            w = weights.unsqueeze(-1) + 1e-8  # (N, V, J, 1)
            pooled = (feat * w).sum(dim=2) / w.sum(dim=2)
        else:
            pooled = feat.mean(dim=2)
        return pooled

    def _features_from_x(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build a per-view descriptor from 2D observations and intrinsics.

        Returns (N, V, 8): mean(x), mean(y), mean(c), cx, cy, fx, fy, skew.
        """
        import sys
        N, V, J, _ = x.shape
        points = x[..., :2]  # (N, V, J, 2)
        conf = x[..., 2]  # (N, V, J)

        # Occluded/padded joints may be NaN.  Mask them out and zero their
        # confidence so they do not contribute to the per-view descriptor.
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(conf)  # (N, V, J)
        points = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
        conf = torch.where(valid, conf, torch.zeros_like(conf))

        if weights is not None:
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            w = (weights * conf).unsqueeze(-1) + 1e-8
            p_mean = (points * w).sum(dim=2) / w.sum(dim=2)
        else:
            p_mean = points.mean(dim=2)

        # Intrinsic parameters per view.
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        skew = K[..., 0, 1]

        feat = torch.stack([p_mean[..., 0], p_mean[..., 1], conf.mean(dim=-1), cx, cy, fx, fy, skew], dim=-1)
        feat = torch.where(torch.isfinite(feat), feat, torch.zeros_like(feat))

        out = self.fallback_projector(feat)
        wb = self.fallback_projector.weight
        # print(f"[DEBUG _features_from_x] points finite={torch.isfinite(points).all().item()} conf finite={torch.isfinite(conf).all().item()} weights finite={torch.isfinite(weights).all().item() if weights is not None else 'n/a'} p_mean finite={torch.isfinite(p_mean).all().item()} feat finite={torch.isfinite(feat).all().item()} out finite={torch.isfinite(out).all().item()} weight finite={torch.isfinite(wb).all().item()}", file=sys.stderr, flush=True)

        return out


def _make_toy_intrinsics(V: int = 4) -> torch.Tensor:
    """Helper for the smoke test."""
    import numpy as np
    K = torch.eye(3).float().unsqueeze(0).repeat(V, 1, 1)
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    return K


if __name__ == "__main__":
    torch.manual_seed(0)

    N, V, J, d = 2, 4, 17, 64
    K = _make_toy_intrinsics(V).unsqueeze(0).expand(N, -1, -1, -1)
    feat = torch.randn(N, V, J, d)
    weights = torch.rand(N, V, J)

    layer = PrincipalPointCorrection(d=d, hidden=64, max_offset=20.0)
    K_corr, delta = layer(K, feat=feat, weights=weights)

    assert K_corr.shape == K.shape
    assert delta.shape == (N, V, 2)
    assert (delta.abs() <= 20.0 + 1e-5).all()

    # Initial offsets should be very small because tanh(0)=0.
    print(f"mean predicted offset at init: {delta.abs().mean().item():.4f} px")
    print("principal-point correction layer smoke test passed")
