"""Intrinsic correction layer for calibrated multi-view pose.

Learns a bounded per-view correction of the intrinsic matrix ``K``:

*   principal-point shift ``(dx, dy)`` in pixels,
*   focal-length scale ``s`` so that ``fx' = s * fx`` and ``fy' = s * fy``.

The corrections are predicted from raw 2D observations + intrinsics and are
kept near the identity at initialization so the layer is transparent when
calibration is accurate.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class IntrinsicCorrection(nn.Module):
    """Learn bounded per-view corrections for principal point and focal length.

    Parameters
    ----------
    d:
        Feature dimension when ``feat`` is provided.
    hidden:
        Hidden size of the correction predictor.
    max_offset:
        Maximum absolute principal-point correction in pixels.
    max_focal_scale:
        Maximum relative focal-length correction, e.g. 0.1 means ``fx, fy``
        are corrected by at most ±10%.
    """

    def __init__(
        self,
        d: int = 64,
        hidden: int = 64,
        max_offset: float = 20.0,
        max_focal_scale: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.max_offset = max_offset
        self.max_focal_scale = max_focal_scale

        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),  # dx, dy, focal log-scale
            nn.Tanh(),
        )

        # Fallback projector for the raw-observation path (8-D descriptor -> d).
        self.fallback_projector = nn.Linear(8, d)

    def forward(
        self,
        K: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict intrinsic corrections and apply them to ``K``.

        Parameters
        ----------
        K:
            Intrinsic matrices ``(N, V, 3, 3)``.
        feat:
            Optional per-view per-joint features ``(N, V, J, d)``.
        x:
            Optional raw input ``(N, V, J, 3)`` with ``[..., :2]`` the 2D
            keypoints and ``[..., 2]`` the confidence.
        weights:
            Optional pooling weights ``(N, V, J)``.

        Returns
        -------
        K_corrected:
            Corrected intrinsic matrices, same shape as ``K``.
        pp_delta:
            Predicted principal-point offsets ``(N, V, 2)`` in pixels.
        focal_scale:
            Predicted focal-length scales ``(N, V)``.
        """
        if feat is not None:
            pooled = self._pool_features(feat, weights)
        elif x is not None:
            pooled = self._features_from_x(x, K, weights)
        else:
            raise ValueError("Either feat or x must be provided.")

        out = self.mlp(pooled)  # (N, V, 3)
        pp_delta = out[..., :2] * self.max_offset  # (N, V, 2)
        # focal_scale ∈ [1 - max_focal_scale, 1 + max_focal_scale]
        focal_scale = 1.0 + out[..., 2] * self.max_focal_scale  # (N, V)

        K_corrected = K.clone()
        K_corrected[..., 0, 2] = K_corrected[..., 0, 2] + pp_delta[..., 0]
        K_corrected[..., 1, 2] = K_corrected[..., 1, 2] + pp_delta[..., 1]
        K_corrected[..., 0, 0] = K_corrected[..., 0, 0] * focal_scale
        K_corrected[..., 1, 1] = K_corrected[..., 1, 1] * focal_scale

        return K_corrected, pp_delta, focal_scale

    def _pool_features(
        self,
        feat: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if weights is not None:
            w = weights.unsqueeze(-1) + 1e-8
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
        N, V, J, _ = x.shape
        points = x[..., :2]
        conf = x[..., 2]

        if weights is not None:
            w = (weights * conf).unsqueeze(-1) + 1e-8
            p_mean = (points * w).sum(dim=2) / w.sum(dim=2)
        else:
            p_mean = points.mean(dim=2)

        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        skew = K[..., 0, 1]

        feat = torch.stack(
            [p_mean[..., 0], p_mean[..., 1], conf.mean(dim=-1), cx, cy, fx, fy, skew],
            dim=-1,
        )
        return self.fallback_projector(feat)
