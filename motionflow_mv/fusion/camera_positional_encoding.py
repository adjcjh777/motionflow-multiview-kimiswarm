"""Geometry-based camera positional encoding.

Replaces dataset-specific learned view embeddings with a deterministic,
cross-camera-rig positional encoding built from calibrated camera parameters.
This lets the same model accept a variable number of views and transfer across
datasets with different camera setups.
"""

import torch
import torch.nn as nn


class CameraPositionalEncoding(nn.Module):
    """Camera positional encoding from intrinsics and extrinsics.

    For each view we compute:
        * camera center in world coordinates: c = -R^T t
        * principal ray direction in world coordinates: r = R^T [0, 0, 1]
        * mean focal length: f = (f_x + f_y) / 2

    These are normalized to be invariant to absolute scene scale and focal
    length, then encoded with Fourier sinusoidal bands and an MLP.

    Parameters
    ----------
    d:
        Output dimension per camera token.
    n_bands:
        Number of Fourier frequency bands (default 4).  Each band contributes
        two channels (sin/cos), so the Fourier feature has
        ``3 * (1 + 2 * n_bands)`` channels for the camera center plus
        ``3 * (1 + 2 * n_bands)`` for the ray direction plus
        ``1 + 2 * n_bands`` for the focal length.
    hidden_dim:
        Hidden dimension of the output MLP.
    """

    def __init__(self, d: int = 64, n_bands: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        self.n_bands = n_bands

        self.fourier_dim = (1 + 2 * n_bands)
        input_dim = 3 * self.fourier_dim + 3 * self.fourier_dim + self.fourier_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )

    def _fourier(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding to the last dimension.

        Args:
            x: tensor of shape (*, C), where values are assumed to be in a
               normalized range (e.g., [-1, 1] or [-0.5, 0.5]).

        Returns:
            Encoded tensor of shape (*, C * (1 + 2 * n_bands)).
        """
        bands = [x]
        for i in range(self.n_bands):
            freq = 2.0 ** i
            bands.append(torch.sin(freq * torch.pi * x))
            bands.append(torch.cos(freq * torch.pi * x))
        return torch.cat(bands, dim=-1)

    def forward(self, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute camera positional encoding.

        Args:
            K: (..., V, 3, 3) intrinsics.
            R: (..., V, 3, 3) rotation matrices.
            t: (..., V, 3) translation vectors.

        Returns:
            Camera tokens of shape (..., V, d).
        """
        # Camera center and principal ray.
        centers = -torch.einsum("...vij,...vj->...vi", R.transpose(-2, -1), t)  # (..., V, 3)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype)
        rays = torch.einsum("...vij,j->...vi", R.transpose(-2, -1), z_axis)  # (..., V, 3)

        # Mean focal length.
        fx = K[..., 0, 0]  # (..., V)
        fy = K[..., 1, 1]
        f = (fx + fy) * 0.5  # (..., V)

        # Normalize per rig to be scale-invariant.
        mean_center = centers.mean(dim=-2, keepdim=True)  # (..., 1, 3)
        scale = torch.norm(centers - mean_center, dim=-1, keepdim=True).max(dim=-2, keepdim=True).values  # (..., 1, 1)
        centers_norm = (centers - mean_center) / (scale + 1e-6)
        rays_norm = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-6)
        f_mean = f.mean(dim=-1, keepdim=True)
        f_norm = f / (f_mean + 1e-6) - 1.0

        # Fourier encoding.
        centers_enc = self._fourier(centers_norm)  # (..., V, 3 * (1 + 2*n_bands))
        rays_enc = self._fourier(rays_norm)
        f_enc = self._fourier(f_norm.unsqueeze(-1))  # (..., V, 1 + 2*n_bands)

        tokens = torch.cat([centers_enc, rays_enc, f_enc], dim=-1)
        return self.mlp(tokens)
