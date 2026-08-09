"""v45-AGF: Adaptive Geometry Fusion for v25 multi-view triangulation.

A lightweight learnable module that predicts per-view (and optionally per-joint)
reliability weights from reprojection residuals.  When ``use_v45`` is enabled
inside ``MultiViewGeometryFusionV25``, the predicted weights multiply the
incoming confidences before the DLT triangulation step, so unreliable views
contribute less.

The module is intentionally small: a tiny MLP over residual statistics.  The
final layer is initialised to zero so that ``2 * sigmoid(0) = 1.0``; this
gives an identity-like behaviour at the start of training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_projection_matrix(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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


class AdaptiveGeometryFusionV45(nn.Module):
    """Predict adaptive reliability weights for weighted DLT triangulation.

    Parameters
    ----------
    n_views:
        Number of camera views (used for shape hints, not for learned buffers).
    weight_type:
        One of ``"per_view"``, ``"per_joint"``, or ``"per_view_joint"``.
        ``per_view`` predicts a single weight per view (shared across joints);
        ``per_joint`` predicts a weight per joint (shared across views? no, per
        view/joint); ``per_view_joint`` predicts a weight per (view, joint).
    hidden:
        Hidden dimension of the MLP.
    n_layers:
        Number of linear layers in the MLP. ``n_layers=1`` is a single linear
        projection to the output weight.
    dropout:
        Dropout probability between MLP layers.
    """

    def __init__(
        self,
        n_views: int = 4,
        weight_type: str = "per_view",
        hidden: int = 32,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_views = n_views
        self.weight_type = weight_type
        self.hidden = hidden
        self.n_layers = max(1, n_layers)
        self.dropout = dropout

        if weight_type not in ("per_view", "per_joint", "per_view_joint"):
            raise ValueError(
                f"Unsupported v45 weight_type: {weight_type}. "
                "Choose from per_view, per_joint, per_view_joint."
            )

        # Input feature dimension:
        #   reprojection residual, log(residual + eps)
        in_dim = 2

        # per_view_joint uses an extra pooled-view feature as well.
        if weight_type == "per_view_joint":
            in_dim = 2 + 2

        layers: list[nn.Module] = []
        for i in range(self.n_layers):
            is_last = i == self.n_layers - 1
            layer_in = in_dim if i == 0 else hidden
            layer_out = 1 if is_last else hidden
            layers.append(nn.Linear(layer_in, layer_out))
            if not is_last:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        # Initialise final layer to zero so the initial weight is ~1.0.
        # We apply 2 * sigmoid to the output, so zero bias -> weight = 1.
        with torch.no_grad():
            final_linear = self.mlp[-1]
            assert isinstance(final_linear, nn.Linear)
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _reprojection_residual(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-(view, joint) reprojection residual norm.

        Args:
            points_2d: (B, T, V, J, 2).
            pred_3d: (B, T, J, 3).
            K: (B, T, V, 3, 3).
            R: (B, T, V, 3, 3).
            t: (B, T, V, 3).

        Returns:
            residual: (B, T, V, J).
        """
        B, T, V, J, _ = points_2d.shape
        X = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B, T, V, J, 3)
        X = X.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
        X_cam = torch.matmul(R, X) + t[..., None]  # (B, T, V, 3, J)
        X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
        Z = X_cam[..., 2:3].clamp(min=1e-6)
        X_norm = X_cam / Z
        uv = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)
        uv = uv[..., :2] / uv[..., 2:3]
        residual = (uv - points_2d).norm(dim=-1)  # (B, T, V, J)
        return residual

    def forward(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict adaptive reliability weights.

        Args:
            points_2d: (B, T, V, J, 2).
            pred_3d: (B, T, J, 3). Initial 3D estimate used to compute residuals.
            K: (B, T, V, 3, 3).
            R: (B, T, V, 3, 3).
            t: (B, T, V, 3).
            view_mask: optional (B, T, V) bool/Float. Masked views get weight 0.

        Returns:
            weights: (B, T, V, J) positive reliability weights. Initialised to
                be ~1.0.
        """
        B, T, V, J, _ = points_2d.shape
        residual = self._reprojection_residual(points_2d, pred_3d, K, R, t)
        residual = residual.clamp(min=1e-6)
        log_residual = torch.log(residual)

        if self.weight_type == "per_view":
            # Pool over joints, then predict a single weight per view.
            feat = torch.stack(
                [
                    residual.mean(dim=-1, keepdim=True),  # (B, T, V, 1)
                    log_residual.mean(dim=-1, keepdim=True),
                ],
                dim=-1,
            )  # (B, T, V, 1, 2)
            feat = feat.expand(-1, -1, -1, J, -1)  # (B, T, V, J, 2)
        elif self.weight_type == "per_joint":
            feat = torch.stack([residual, log_residual], dim=-1)  # (B, T, V, J, 2)
        else:  # per_view_joint
            # Combine per-joint residual with pooled view-level statistics.
            pooled_mean = residual.mean(dim=-1, keepdim=True).expand(-1, -1, -1, J)
            pooled_log = log_residual.mean(dim=-1, keepdim=True).expand(-1, -1, -1, J)
            feat = torch.stack(
                [residual, log_residual, pooled_mean, pooled_log],
                dim=-1,
            )  # (B, T, V, J, 4)

        weights = self.mlp(feat).squeeze(-1)  # (B, T, V, J)
        weights = 2.0 * torch.sigmoid(weights)

        if view_mask is not None:
            mask = view_mask.float().unsqueeze(-1)  # (B, T, V, 1)
            weights = weights * mask

        return weights.clamp(min=1e-4, max=1e4)
