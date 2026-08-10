"""v79 Canonical View Refinement (CVR).

A lightweight geometric refinement head inspired by ARGUS-style canonical-view
feature warping.  It projects the current 3-D pose estimate into a synthetic
front-facing canonical camera, feeds the resulting canonical 2-D/depth
features through a small MLP, and adds a gated residual to the input pose.

The module is intentionally image-free: it only uses the existing 2-D
keypoints and calibrated camera parameters.  At initialization the residual
output layer is zero and the residual gate is closed, so the module is
strictly identity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonicalViewRefinementV79(nn.Module):
    """Canonical-view geometric refinement head.

    Parameters
    ----------
    j:
        Number of joints.
    hidden:
        Hidden dimension of the residual MLP.
    n_layers:
        Total number of MLP layers (input + hidden + output).  The network
        consists of an first ``Linear(in_dim, hidden)``, ``n_layers - 1``
        hidden ``Linear(hidden, hidden)`` layers, and a final
        ``Linear(hidden, 3)`` that produces the residual.
    identity_init:
        Zero-initialize the final residual projection so the output equals
        the input at model initialization.
    residual_gate_init:
        Initial value of the scalar residual gate logit.  A value of ``-6.0``
        gives ``sigmoid(gate) ≈ 0.0025``.
    """

    def __init__(
        self,
        j: int = 17,
        hidden: int = 64,
        n_layers: int = 2,
        identity_init: bool = True,
        residual_gate_init: float = -6.0,
    ) -> None:
        super().__init__()
        self.j = j
        self.hidden = hidden
        self.n_layers = max(1, n_layers)
        self.identity_init = identity_init

        in_dim = 3 + 2 + 1  # 3D pose + canonical 2D + canonical depth
        self.first = nn.Linear(in_dim, hidden)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden, hidden))
        self.residual_out = nn.Linear(hidden, 3)

        if self.identity_init:
            nn.init.zeros_(self.residual_out.weight)
            nn.init.zeros_(self.residual_out.bias)

        # Scalar residual gate.  Small sigmoid at init keeps the module identity.
        self.gate_logit = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float))

    def _canonical_camera(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple:
        """Build a canonical front-facing camera from the input cameras.

        Args:
            K: ``(B, T, V, 3, 3)`` intrinsics.
            R: ``(B, T, V, 3, 3)`` rotations matrices.
            t: ``(B, T, V, 3)`` translation vectors.

        Returns:
            K_can: ``(B, T, 3, 3)`` averaged intrinsics.
            R_can: ``(3, 3)`` identity rotation.
            t_can: ``(B, T, 3)`` canonical translation placing the camera on the
                +Z axis at the mean camera depth.
        """
        B, T, V, *_ = K.shape
        # Average intrinsics over views.
        K_can = K.mean(dim=2)  # (B, T, 3, 3)

        # Front-facing, Y-up canonical rotation.
        R_can = torch.eye(3, device=K.device, dtype=K.dtype)

        # Mean camera depth; clamp to avoid degenerate canonical camera.
        z0 = t.norm(dim=-1).mean()
        z0 = z0.clamp(min=1e-3)
        t_can = torch.zeros(B, T, 3, device=K.device, dtype=K.dtype)
        t_can[..., 2] = -z0

        return K_can, R_can, t_can

    def _project_canonical(
        self,
        pred_3d: torch.Tensor,
        K_can: torch.Tensor,
        R_can: torch.Tensor,
        t_can: torch.Tensor,
    ) -> tuple:
        """Project the pose into the canonical camera and normalize coordinates.

        Args:
            pred_3d: ``(B, T, J, 3)`` pose.
            K_can: ``(B, T, 3, 3)`` canonical intrinsics.
            R_can: ``(3, 3)`` canonical rotation.
            t_can: ``(B, T, 3)`` canonical translation.

        Returns:
            x_can_norm: ``(B, T, J, 2)`` normalized canonical 2-D coordinates.
            z_can_norm: ``(B, T, J, 1)`` normalized canonical depth.
        """
        # Center the root joint at the origin.
        root = pred_3d.mean(dim=-2, keepdim=True)  # (B, T, 1, 3)
        X = pred_3d - root  # (B, T, J, 3)

        # Camera-space coordinates: X_cam = R_can @ X + t_can.
        # R_can is identity, so a simple broadcast add is sufficient.
        t = t_can[:, :, None, :]  # (B, T, 1, 3)
        X_cam = X + t  # (B, T, J, 3)

        # Project: x_h = K_can @ X_cam^T per joint.
        # K_can is (B, T, 3, 3), X_cam is (B, T, J, 3).
        X_h = torch.einsum("btij,btvj->btvi", K_can, X_cam)  # (B, T, J, 3)

        z = X_h[..., 2:]  # (B, T, J, 1)
        x = X_h[..., :2] / (z + 1e-6)  # (B, T, J, 2)

        # Normalize by focal length and principal offset.
        fx = K_can[..., 0, 0]  # (B, T)
        fy = K_can[..., 1, 1]
        cx = K_can[..., 0, 2]
        cy = K_can[..., 1, 2]
        x_norm = torch.stack(
            [
                (x[..., 0] - cx[..., None]) / (fx[..., None] + 1e-6),
                (x[..., 1] - cy[..., None]) / (fy[..., None] + 1e-6),
            ],
            dim=-1,
        )  # (B, T, J, 2)

        # Normalize depth by the canonical camera distance.
        z_norm = z / (z.abs().mean(dim=-2, keepdim=True) + 1e-6)

        return x_norm, z_norm

    def forward(
        self,
        pred_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine ``pred_3d`` with canonical-view geometric features.

        Args:
            pred_3d: ``(B, T, J, 3)`` current 3-D pose estimate.
            K: ``(B, T, V, 3, 3)`` camera intrinsics.
            R: ``(B, T, V, 3, 3)`` camera rotations matrices.
            t: ``(B, T, V, 3)`` camera translation vectors.
            view_mask: Optional ``(B, T, V)`` binary view mask (unused).

        Returns:
            ``(B, T, J, 3)`` refined 3-D pose.
        """
        K_can, R_can, t_can = self._canonical_camera(K, R, t)
        x_can, z_can = self._project_canonical(pred_3d, K_can, R_can, t_can)

        feat = torch.cat([pred_3d, x_can, z_can], dim=-1)  # (B, T, J, 6)

        h = F.relu(self.first(feat))
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        residual = self.residual_out(h)  # (B, T, J, 3)

        gate = torch.sigmoid(self.gate_logit)
        return pred_3d + gate * residual
