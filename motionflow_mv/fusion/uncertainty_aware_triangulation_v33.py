"""v33 uncertainty-aware triangulation head.

Predicts per-view, per-joint 2-D log-variance from spatio-temporal features,
re-weights the DLT triangulation with the resulting diagonal precision, and
supervises the predicted uncertainty with a reprojection negative-log-likelihood
loss.  The module is optional and identity at init: the residual refinement MLP
is gated by a learnable scalar initialised to 0, so the baseline is preserved when
the flag is off.
"""

from typing import Optional

import torch
import torch.nn as nn


def _triangulate_precision_weighted_batched(
    points_2d: torch.Tensor,
    proj_matrices: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """Batched DLT triangulation with per-axis precision weights.

    Args:
        points_2d: (N, V, J, 2)
        proj_matrices: (N, V, 3, 4)
        precision: (N, V, J, 2) positive per-axis precision (1 / variance).

    Returns:
        X: (N, J, 3)
    """
    N, V, J, _ = points_2d.shape
    if proj_matrices.dim() == 3:
        proj_matrices = proj_matrices.unsqueeze(0).expand(N, -1, -1, -1)

    sqrt_precision = torch.sqrt(precision + 1e-8)  # (N, V, J, 2)

    A_rows = []
    for v in range(V):
        P = proj_matrices[:, v, :, :]  # (N, 3, 4)
        x = points_2d[:, v, :, 0]  # (N, J)
        y = points_2d[:, v, :, 1]
        # Each DLT row is a 4-vector.
        row_x = x[..., None] * P[:, 2:3, :] - P[:, 0:1, :]  # (N, J, 4)
        row_y = y[..., None] * P[:, 2:3, :] - P[:, 1:2, :]

        w_x = sqrt_precision[:, v, :, 0].unsqueeze(-1)  # (N, J, 1)
        w_y = sqrt_precision[:, v, :, 1].unsqueeze(-1)
        A_view = torch.stack([row_x * w_x, row_y * w_y], dim=2)  # (N, J, 2, 4)
        A_rows.append(A_view)

    A = torch.cat(A_rows, dim=2)  # (N, J, 2V, 4)
    A3 = A[..., :3]  # (N, J, 2V, 3)
    b = -A[..., 3:]  # (N, J, 2V, 1)

    X, *_ = torch.linalg.lstsq(A3, b)
    return X.squeeze(-1)  # (N, J, 3)


class _UncertaintyMLP(nn.Module):
    """Predict per-view, per-joint 2-D log-variance from per-joint tokens."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, V, J, d)
        Returns:
            log_var: (B, T, V, J, 2)
        """
        return self.net(x)


class UncertaintyAwareTriangulationV33(nn.Module):
    """Uncertainty-aware triangulation head.

    Parameters
    ----------
    d:
        Feature dimension of the per-view, per-joint input tokens.
    covariance_hidden:
        Hidden dimension of the 2-layer log-variance prediction MLP.
    log_var_min, log_var_max:
        Clamping range for the predicted log-variance.
    """

    def __init__(
        self,
        d: int,
        covariance_hidden: int = 64,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()
        self.d = d
        self.covariance_hidden = covariance_hidden
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.uncertainty_mlp = _UncertaintyMLP(d, covariance_hidden)

        # Tiny residual MLP around the initial estimate.  Identity-at-init is
        # enforced by the scale parameter below, which starts at 0.
        self.residual_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        points_2d: torch.Tensor,
        confidences: torch.Tensor,
        features: torch.Tensor,
        proj_matrices: torch.Tensor,
        pred_3d_init: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            points_2d: (B, T, V, J, 2)
            confidences: (B, T, V, J)
            features: (B, T, V, J, d)
            proj_matrices: (B, T, V, 3, 4)
            pred_3d_init: (B, T, J, 3)
            view_mask: optional (B, T, V)

        Returns:
            pred_3d_ref: (B, T, J, 3)
            uncertainty_loss: scalar
        """
        B, T, V, J, _ = points_2d.shape
        N = B * T

        # 1. Predict per-view, per-joint 2-D log-variance.
        log_var = self.uncertainty_mlp(features)  # (B, T, V, J, 2)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)

        # Convert to variance.  Add a small floor so division is safe.
        var = torch.exp(log_var) + 1e-8  # (B, T, V, J, 2)

        # Mask out unavailable views by setting variance very high so they
        # contribute almost nothing to the weighted triangulation.
        if view_mask is not None:
            var = torch.where(
                view_mask.view(B, T, V, 1, 1).bool(),
                var,
                torch.full_like(var, 1e6),
            )

        # Flatten batch and time for the batched triangulation helper.
        points_flat = points_2d.view(N, V, J, 2)
        conf_flat = confidences.view(N, V, J)
        proj_flat = proj_matrices.view(N, V, 3, 4)
        var_flat = var.view(N, V, J, 2)

        # 2. Precision-weighted triangulation.
        # Per-axis precision = confidence / variance.
        precision = conf_flat.unsqueeze(-1) / var_flat  # (N, V, J, 2)
        pred_weighted = _triangulate_precision_weighted_batched(
            points_2d=points_flat,
            proj_matrices=proj_flat,
            precision=precision,
        )  # (N, J, 3)
        pred_weighted = pred_weighted.view(B, T, J, 3)

        # 3. Identity-at-init residual refinement.
        delta = pred_weighted - pred_3d_init
        residual = self.residual_mlp(delta)  # (B, T, J, 3)
        pred_3d_ref = pred_3d_init + self.residual_scale * residual

        # 4. Reprojection NLL loss on views with positive confidence.
        uncertainty_loss = self.reprojection_nll(
            pred_3d_ref,
            points_2d,
            confidences,
            proj_matrices,
            var,
        )

        return pred_3d_ref, uncertainty_loss

    def reprojection_nll(
        self,
        pred_3d: torch.Tensor,
        points_2d: torch.Tensor,
        confidences: torch.Tensor,
        proj_matrices: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reprojection negative log-likelihood.

        Loss = 0.5 * (r^T Σ^{-1} r + log det Σ) averaged over views with
        positive confidence.  For a diagonal covariance with axes var_x, var_y,
        this reduces to ``sum(r_i^2 / var_i) + log(var_x * var_y)``.
        """
        # Reproject pred_3d into every view.
        uv_pred = self._project(pred_3d, proj_matrices)  # (B, T, V, J, 2)

        r = uv_pred - points_2d  # (B, T, V, J, 2)
        valid = confidences > 0  # (B, T, V, J)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)

        # Diagonal precision-weighted squared residual.
        precision_residual = (r ** 2) / (var + 1e-8)  # (B, T, V, J, 2)
        quadratic = precision_residual.sum(dim=-1)  # (B, T, V, J)

        log_det = torch.log(var[..., 0] + 1e-8) + torch.log(var[..., 1] + 1e-8)

        nll = 0.5 * (quadratic + log_det)

        # Only average over valid (positive confidence) observations.
        nll = nll[valid].mean()
        return nll

    @staticmethod
    def _project(
        points_3d: torch.Tensor, proj_matrices: torch.Tensor
    ) -> torch.Tensor:
        """Project 3-D points through calibrated cameras.

        Args:
            points_3d: (B, T, J, 3)
            proj_matrices: (B, T, V, 3, 4)

        Returns:
            uv: (B, T, V, J, 2)
        """
        B, T, V, _, _ = proj_matrices.shape
        # Convert to homogeneous.
        X_h = torch.cat(
            [points_3d, torch.ones(B, T, points_3d.shape[2], 1, device=points_3d.device, dtype=points_3d.dtype)],
            dim=-1,
        )  # (B, T, J, 4)
        # Repeat over view dimension.
        X_h = X_h.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B, T, V, J, 4)
        x_h = torch.einsum("btvij,btvkj->btvki", proj_matrices, X_h)
        uv = x_h[..., :2] / (x_h[..., 2:3] + 1e-8)
        return uv
