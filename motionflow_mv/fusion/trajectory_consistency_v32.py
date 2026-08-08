"""Trajectory-Consistency Refiner (v32).

A tiny 1-D temporal CNN over the predicted 3-D pose sequence, plus a
trajectory-consistency loss with a smoothness and a drift-guard term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryConsistencyRefinerV32(nn.Module):
    """Lightweight temporal smoother operating on a pose sequence.

    Input shape: (B, T, J, 3)
    Output shape: (B, T, J, 3)

    The module is initialized as a no-op: the final conv layer is zero-initialised
    and the residual gate is zero, so the refined pose equals the input at the
    start of training.
    """

    def __init__(self, n_joints: int, kernel_size: int = 5, hidden_channels: int = 64) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        super().__init__()
        self.n_joints = n_joints
        self.pad = kernel_size // 2
        in_ch = n_joints * 3
        self.conv1 = nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=self.pad)
        self.conv2 = nn.Conv1d(hidden_channels, in_ch, kernel_size, padding=self.pad)
        # Initialize final layer to zero so the residual delta is zero at start.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        # Gate initialized to zero -> no-op at start of training.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, J, 3). Returns: refined (B, T, J, 3)."""
        B, T, J, _ = x.shape
        # (B, T, J*3) -> (B, J*3, T)
        x_flat = x.reshape(B, T, J * 3).transpose(1, 2)
        h = F.relu(self.conv1(x_flat))
        delta = self.conv2(h)
        delta = delta.transpose(1, 2).reshape(B, T, J, 3)
        return x + self.gate * delta


def trajectory_consistency_loss(
    refined: torch.Tensor, raw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (smoothness_loss, drift_loss).

    smoothness_loss: mean squared second-order finite difference along time.
    drift_loss: L2 distance between refined and raw trajectories.
    """
    # Smoothness via second difference: x[t-1] - 2x[t] + x[t+1]
    second_diff = refined[:, 1:-1] - 2.0 * refined[:, 1:-1] + refined[:, 1:-1]
    # Use central finite difference explicitly.
    second_diff = refined[:, :-2] - 2.0 * refined[:, 1:-1] + refined[:, 2:]
    smoothness = (second_diff ** 2).mean()
    drift = ((refined - raw) ** 2).mean()
    return smoothness, drift
