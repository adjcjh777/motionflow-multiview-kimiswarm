"""Rotation correction head for calibrated multi-view pose.

Predicts a bounded SO(3) residual per view from pooled per-view features and
applies it to the camera rotation matrix before triangulation.  The residual
is parameterised as an axis-angle vector, squashed with ``tanh`` so that the
layer is initialised near the identity and cannot catastrophically rotate
the rig.
"""

from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn


class RotationCorrectionHead(nn.Module):
    """Predict a bounded SO(3) residual rotation per view.

    The module accepts per-view features, predicts a 3-D axis-angle vector for
    each view, bounds the magnitude with ``tanh``, converts the vector to an
    SO(3) matrix via the matrix exponential, and applies it to the input
    rotation matrices.

    Parameters
    ----------
    d:
        Feature dimension.
    hidden:
        Hidden size of the axis-angle predictor.
    max_rot_deg:
        Maximum absolute value (in degrees) for each axis-angle component.
        The resulting rotation angle is therefore bounded by
        ``sqrt(3) * max_rot_deg``.
    """

    def __init__(self, d: int = 64, hidden: int = 64, max_rot_deg: float = 2.0):
        super().__init__()
        self.d = d
        self.max_rot_deg = max_rot_deg
        self.max_rot_rad = math.radians(max_rot_deg)

        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

        # Initialise the final layer to zero so the residual is the identity
        # matrix at the beginning of training.  Earlier layers keep their
        # default initialisation so gradients can flow.
        self._zero_init_last_linear(self.mlp[-1])

    @staticmethod
    def _zero_init_last_linear(layer: nn.Linear) -> None:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    def forward(
        self,
        feat: torch.Tensor,
        R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict and apply a residual rotation to ``R``.

        Parameters
        ----------
        feat:
            Per-view features.  Either ``(N, V, d)`` already pooled, or
            ``(N, V, J, d)`` in which case features are mean-pooled over joints.
        R:
            Input rotation matrices ``(N, V, 3, 3)``.

        Returns
        -------
        R_corrected:
            ``(N, V, 3, 3)`` corrected rotation matrices.
        delta_R:
            ``(N, V, 3, 3)`` residual rotation matrices such that
            ``R_corrected = delta_R @ R``.
        """
        if feat.dim() == 4:
            feat = feat.mean(dim=2)
        if feat.dim() != 3:
            raise ValueError(
                f"feat must be (N, V, d) or (N, V, J, d), got shape {tuple(feat.shape)}"
            )

        N, V, d = feat.shape
        r = self.mlp(feat.view(N * V, d)).view(N, V, 3)
        # Bound each axis-angle component to [-max_rot_rad, +max_rot_rad].
        r = torch.tanh(r) * self.max_rot_rad

        delta_R = self._so3_from_rvec(r)
        R_corrected = torch.einsum("nvij,nvjk->nvik", delta_R, R)
        return R_corrected, delta_R

    @staticmethod
    def _so3_from_rvec(rvec: torch.Tensor) -> torch.Tensor:
        """Convert a batch of axis-angle vectors to SO(3) matrices.

        Uses the matrix exponential of the skew-symmetric matrix, which is
        differentiable and exact.
        """
        N, V, _ = rvec.shape
        r_flat = rvec.view(N * V, 3)
        K = _skew(r_flat)
        R = torch.linalg.matrix_exp(K)
        return R.view(N, V, 3, 3)


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrices for a batch of 3-D vectors."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _geodesic_angle(R: torch.Tensor) -> torch.Tensor:
    """Return the geodesic angle (radians) of one or more rotation matrices."""
    trace = torch.einsum("...ii->...", R).clamp(-3.0, 3.0)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


def _make_toy_rotations(V: int = 4) -> torch.Tensor:
    """Build a batch of simple camera rotations for smoke tests."""
    import numpy as np

    R_list = []
    for i in range(V):
        theta = 2 * np.pi * i / V
        # Camera looks at origin from a circle in the x-y plane.
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        R_list.append(R)
    return torch.from_numpy(np.stack(R_list, axis=0)).float()


if __name__ == "__main__":
    torch.manual_seed(0)

    N, V, d = 2, 4, 64
    feat = torch.randn(N, V, d)
    R = _make_toy_rotations(V).unsqueeze(0).expand(N, -1, -1, -1)

    head = RotationCorrectionHead(d=d, hidden=64, max_rot_deg=2.0)
    R_corrected, delta_R = head(feat, R)

    assert R_corrected.shape == (N, V, 3, 3)
    assert delta_R.shape == (N, V, 3, 3)
    assert torch.isfinite(R_corrected).all()
    assert torch.isfinite(delta_R).all()

    # At init the residual must be identity, so the corrected R equals input R.
    assert torch.allclose(R_corrected, R, atol=1e-6)

    # The residual rotation matrices must be proper rotations.
    I = torch.eye(3, device=delta_R.device, dtype=delta_R.dtype)
    identity_check = torch.einsum("nvij,nvjk->nvik", delta_R, delta_R.transpose(-2, -1))
    assert torch.allclose(identity_check, I, atol=1e-4)
    det = torch.det(delta_R)
    assert ((det - 1.0).abs() < 1e-4).all()

    # Apply a small, known rotation by hand and verify it is bounded.
    with torch.no_grad():
        head.mlp[-1].bias[0] = 0.5  # about one axis, within tanh bounds.
    R_corrected2, delta_R2 = head(feat, R)
    angle = _geodesic_angle(delta_R2)
    max_angle = math.radians(2.0)
    assert (angle <= max_angle + 1e-5).all()
    assert not torch.allclose(R_corrected2, R, atol=1e-6)

    # Gradient sanity check.
    head.zero_grad()
    loss = R_corrected2.sum()
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())

    print("RotationCorrectionHead CPU smoke test passed")
