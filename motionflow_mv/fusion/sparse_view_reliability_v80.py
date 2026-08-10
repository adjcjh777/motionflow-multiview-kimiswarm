"""v80 View-Reliability Before Triangulation (VRBT).

A small standalone learned head that predicts per-view reliability weights
*before* triangulation.  It consumes per-view feature tokens and geometry
cues (reprojection error and an epipolar-style residual), and outputs a
sigmoid-activated reliability in ``(0, 1)``.  The final MLP layer is
zero-initialised so that at training start the reliability is approximately
``0.5`` (identity behaviour) and only deviates once data supports it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseViewReliabilityV80(nn.Module):
    """Predict per-view reliability weights for multi-view pose triangulation.

    Parameters
    ----------
    d:
        Feature dimension of the input tokens.
    n_views:
        Maximum number of camera views (shape hint only).
    n_joints:
        Number of joints (shape hint only).
    hidden:
        Hidden dimension of the MLP.
    weight_type:
        ``per_view`` pools across joints and predicts one weight per view;
        ``per_view_joint`` keeps per-joint predictions.
    """

    def __init__(
        self,
        d: int = 64,
        n_views: int = 4,
        n_joints: int = 17,
        hidden: int = 64,
        n_layers: int = 2,
        weight_type: str = "per_view",
        use_geometry_bias: bool = True,
        use_feature_bias: bool = True,
        identity_init: bool = True,
        min_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if weight_type not in ("per_view", "per_view_joint"):
            raise ValueError(
                f"weight_type must be 'per_view' or 'per_view_joint', got {weight_type}"
            )

        self.d = d
        self.n_views = n_views
        self.n_joints = n_joints
        self.hidden = hidden
        self.weight_type = weight_type
        self.min_weight = min_weight

        # Input: pooled/raw features (d) + reprojection error (1) + epipolar residual (1)
        layers: list[nn.Module] = [
            nn.Linear(d + 2, hidden),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

        # Optionally zero-initialise the final layer so reliability starts ~0.5 after sigmoid.
        final_linear = self.mlp[-1]
        assert isinstance(final_linear, nn.Linear)
        if identity_init:
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _project(
        self,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Project world 3D points into each view. Returns (B, T, V, J, 2)."""
        # pred_3d: (B, T, J, 3) -> (B, T, 1, J, 3, 1) broadcast over views.
        X = pred_3d.unsqueeze(2).unsqueeze(-1)
        # R: (B, T, V, 3, 3) -> (B, T, V, 1, 3, 3) broadcast over joints.
        R = R.unsqueeze(3)
        t = t.unsqueeze(-2)  # (B, T, V, 1, 3)
        X_cam = (R @ X).squeeze(-1) + t  # (B, T, V, J, 3)

        # Project with intrinsics: uv = K @ X_cam / z
        z = X_cam[..., 2:3].clamp(min=1e-6)
        proj = (K.unsqueeze(3) @ X_cam.unsqueeze(-1)).squeeze(-1)  # (B, T, V, J, 3)
        uv = proj[..., :2] / z
        return uv

    def _reprojection_error(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-view reprojection error. Returns (B, T, V, J)."""
        uv = self._project(pred_3d, K, R, t)
        residual = (uv - points_2d).norm(dim=-1)
        return residual

    def _epipolar_residual(self, reproj_error: torch.Tensor) -> torch.Tensor:
        """Per-view deviation from mean reprojection error. Returns (B, T, V, J)."""
        mean_error = reproj_error.mean(dim=2, keepdim=True)  # (B, T, 1, J)
        return reproj_error - mean_error

    def forward(
        self,
        features: torch.Tensor,
        points_2d: torch.Tensor,
        pred_3d: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        pred_3d_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict per-view reliability weights.

        Args
        ----
        features:
            ``(B, T, V, J, d)`` feature tokens.
        points_2d:
            ``(B, T, V, J, 2)`` input 2-D keypoints.
        pred_3d:
            ``(B, T, J, 3)`` initial triangulated 3-D estimate.
        K, R:
            ``(B, T, V, 3, 3)`` camera intrinsics and rotations.
        t:
            ``(B, T, V, 3)`` camera translations.
        view_mask:
            Optional ``(B, T, V)`` bool/float mask. ``True`` / ``1.0`` means valid.

        Returns
        -------
        reliability:
            ``(B, T, V)`` if ``weight_type == 'per_view'``,
            ``(B, T, V, J)`` if ``weight_type == 'per_view_joint'``.
            Values are in ``(0, 1)``.
        """
        B, T, V, J, d = features.shape

        # Accept either pred_3d or the caller-friendly alias pred_3d_init.
        if pred_3d is None:
            pred_3d = pred_3d_init
        if pred_3d is None:
            raise ValueError("Either pred_3d or pred_3d_init must be provided.")

        reproj_error = self._reprojection_error(pred_3d, points_2d, K, R, t)
        epipolar_res = self._epipolar_residual(reproj_error)

        if self.weight_type == "per_view":
            feat = features.mean(dim=3)  # (B, T, V, d)
            reproj_pooled = reproj_error.mean(dim=-1, keepdim=True)  # (B, T, V, 1)
            epipolar_pooled = epipolar_res.mean(dim=-1, keepdim=True)  # (B, T, V, 1)
            x = torch.cat([feat, reproj_pooled, epipolar_pooled], dim=-1)  # (B, T, V, d+2)
            logit = self.mlp(x).squeeze(-1)  # (B, T, V)
        else:  # per_view_joint
            reproj_feat = reproj_error.unsqueeze(-1)  # (B, T, V, J, 1)
            epipolar_feat = epipolar_res.unsqueeze(-1)  # (B, T, V, J, 1)
            x = torch.cat([features, reproj_feat, epipolar_feat], dim=-1)  # (B, T, V, J, d+2)
            logit = self.mlp(x).squeeze(-1)  # (B, T, V, J)

        if view_mask is not None:
            # Apply a large negative bias to masked views so they get ~0 reliability.
            mask = view_mask.float()
            if self.weight_type == "per_view":
                logit = logit + (1.0 - mask) * -1e9
            else:
                logit = logit + (1.0 - mask).unsqueeze(-1) * -1e9

        reliability = torch.sigmoid(logit)
        if self.min_weight > 0.0:
            reliability = reliability.clamp(min=self.min_weight)
        if view_mask is not None:
            if self.weight_type == "per_view":
                reliability = reliability * view_mask.float()
            else:
                reliability = reliability * view_mask.float().unsqueeze(-1)
        return reliability


# Alias used by motionflow_mv/fusion/omniview_fusion_v5.py
ViewReliabilityHeadV80 = SparseViewReliabilityV80
