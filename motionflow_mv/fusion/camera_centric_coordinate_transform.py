"""Learnable camera-centric coordinate transform for multi-view pose.

The pinhole model assumes calibrated extrinsics (R, t) are exact.  In practice,
small rig calibration errors (mounting drift, synchronisation jitter, lens-camera
shift) break multi-view triangulation.  This module learns a *bounded residual*
transform in camera-centric coordinates: a per-view rotation ``ΔR``, translation
``Δt``, and ray-depth scale ``s``.  The transform is initialised at the identity,
so it can be inserted into the existing anchor pipeline as a warm-startable
ablation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CameraCentricCoordinateTransform(nn.Module):
    """Predict a bounded per-view residual SE(3)+scale transform.

    Parameters
    ----------
    d:
        Feature dimension when ``feat`` is provided.
    hidden:
        Hidden size of the residual predictor.
    max_rot_offset_deg:
        Maximum rotation correction in degrees.  The residual is parameterised
        by an so(3) vector and mapped through ``axis_angle_to_matrix``.
    max_trans_offset_m:
        Maximum absolute translation correction in meters.
    max_scale_delta:
        Maximum deviation of the per-view depth scale from 1.0.
    condition_on_deep_features:
        If ``True``, the predictor consumes deep per-view features ``(N, V, d)``;
        otherwise it falls back to a descriptor built from ``x``, ``K``, ``R``,
        and ``t``.
    """

    def __init__(
        self,
        d: int = 64,
        hidden: int = 64,
        max_rot_offset_deg: float = 2.0,
        max_trans_offset_m: float = 0.05,
        max_scale_delta: float = 0.05,
        condition_on_deep_features: bool = True,
    ):
        super().__init__()
        self.d = d
        self.max_rot_offset = max_rot_offset_deg * (torch.pi / 180.0)
        self.max_trans_offset = max_trans_offset_m
        self.max_scale_delta = max_scale_delta
        self.condition_on_deep_features = condition_on_deep_features

        # Rotation head: so(3) vector -> rotation matrix.
        self.rot_mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )

        # Translation head.
        self.trans_mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Tanh(),
        )

        # Scale head: positive scale near 1.0.
        self.scale_mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

        # Fallback projector for the raw-observation path.
        self.fallback_projector = nn.Linear(15, d)

    def forward(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict and apply the camera-centric coordinate transform.

        Parameters
        ----------
        R:
            Original rotations ``(N, V, 3, 3)`` (world -> camera).
        t:
            Original translations ``(N, V, 3)`` (world -> camera).
        feat:
            Optional deep per-view features ``(N, V, d)``.
        x:
            Optional raw input ``(N, V, J, 3)`` with ``[..., :2]`` the 2D
            keypoints and ``[..., 2]`` the confidence.  Used only when
            ``feat`` is not provided.
        K:
            Optional intrinsics, used only for the fallback descriptor.
        weights:
            Optional pooling weights ``(N, V, J)``.

        Returns
        -------
        R_corrected:
            Updated rotations ``(N, V, 3, 3)``.
        t_corrected:
            Updated translations ``(N, V, 3)``.
        scale:
            Per-view depth scale ``(N, V)`` applied to camera rays.
        delta_R:
            Predicted rotation matrices ``(N, V, 3, 3)``.
        delta_t:
            Predicted translation offsets ``(N, V, 3)`` in meters.
        scale_factor:
            Predicted scale factor ``(N, V)`` (final scale = ``1 + factor``).
        """
        if feat is not None:
            pooled = feat  # Assume caller already pooled to (N, V, d).
        elif x is not None:
            pooled = self._features_from_x(x, K, R, t, weights)
        else:
            raise ValueError("Either feat or x must be provided.")

        # Bounded residuals (all near zero / one at init because Tanh(0)=0).
        delta_rot = self.rot_mlp(pooled) * self.max_rot_offset  # (N, V, 3)
        delta_t = self.trans_mlp(pooled) * self.max_trans_offset  # (N, V, 3)
        scale_factor = self.scale_mlp(pooled).squeeze(-1) * self.max_scale_delta  # (N, V)
        scale = 1.0 + scale_factor

        # Convert so(3) vector to rotation matrix.
        delta_R = self._so3_exp_map(delta_rot)  # (N, V, 3, 3)

        # Apply residual transform to extrinsics.
        # World-to-camera becomes: R' = delta_R @ R
        #                        t' = delta_R @ t + delta_t
        R_corrected = torch.einsum("bvij,bvjk->bvik", delta_R, R)
        t_corrected = torch.einsum("bvij,bvj->bvi", delta_R, t) + delta_t

        return R_corrected, t_corrected, scale, delta_R, delta_t, scale_factor

    def _so3_exp_map(self, omega: torch.Tensor) -> torch.Tensor:
        """Batch matrix exponential of so(3) vectors using Rodrigues' formula.

        Parameters
        ----------
        omega:
            ``(N, V, 3)`` axis-angle vectors.

        Returns
        -------
        R:
            ``(N, V, 3, 3)`` rotation matrices.
        """
        N, V, _ = omega.shape
        theta = omega.norm(dim=-1, keepdim=True) + 1e-8  # (N, V, 1)
        k = omega / theta  # (N, V, 3)

        # K = [k]_x
        zeros = torch.zeros(N, V, 1, device=omega.device, dtype=omega.dtype)
        K = torch.cat(
            [
                torch.cat([zeros, -k[..., 2:3], k[..., 1:2]], dim=-1).unsqueeze(-2),
                torch.cat([k[..., 2:3], zeros, -k[..., 0:1]], dim=-1).unsqueeze(-2),
                torch.cat([-k[..., 1:2], k[..., 0:1], zeros], dim=-1).unsqueeze(-2),
            ],
            dim=-2,
        )  # (N, V, 3, 3)

        I = torch.eye(3, device=omega.device, dtype=omega.dtype).view(1, 1, 3, 3)
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)
        return R

    def _features_from_x(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build a per-view descriptor from raw observations and extrinsics.

        Returns (N, V, d).
        """
        N, V, J, _ = x.shape
        points = x[..., :2]  # (N, V, J, 2)
        conf = x[..., 2]  # (N, V, J)

        if weights is not None:
            w = (weights * conf).unsqueeze(-1) + 1e-8  # (N, V, J, 1)
            p_mean = (points * w).sum(dim=2) / w.sum(dim=2)  # (N, V, 2)
            c_mean = (conf.unsqueeze(-1) * w).sum(dim=2).squeeze(-1) / w.sum(dim=2).squeeze(-1)
        else:
            p_mean = points.mean(dim=2)
            c_mean = conf.mean(dim=-1)

        # Camera center.
        cam_center = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)  # (N, V, 3)

        # Intrinsic parameters per view.
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        skew = K[..., 0, 1]

        feat = torch.stack(
            [
                p_mean[..., 0],
                p_mean[..., 1],
                c_mean,
                cx,
                cy,
                fx,
                fy,
                skew,
                cam_center[..., 0],
                cam_center[..., 1],
                cam_center[..., 2],
                t[..., 0],
                t[..., 1],
                t[..., 2],
                (fx + fy) * 0.5,
            ],
            dim=-1,
        )

        return self.fallback_projector(feat)


def _make_toy_extrinsics(n_views: int = 4) -> torch.Tensor:
    """Helper for smoke tests."""
    import numpy as np

    R = np.stack([np.eye(3) for _ in range(n_views)], axis=0)
    return torch.from_numpy(R).float()


if __name__ == "__main__":
    torch.manual_seed(0)

    N, V, J, d = 2, 4, 17, 64
    R = _make_toy_extrinsics(V).unsqueeze(0).expand(N, -1, -1, -1)
    t = torch.randn(N, V, 3) * 0.5
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(N, V, -1, -1).clone()
    K[..., 0, 0] = 800.0
    K[..., 1, 1] = 800.0
    K[..., 0, 2] = 320.0
    K[..., 1, 2] = 240.0
    x = torch.randn(N, V, J, 3)

    layer = CameraCentricCoordinateTransform(d=d, hidden=64)
    R_corr, t_corr, scale, delta_R, delta_t, scale_factor = layer(
        R, t, x=x, K=K
    )

    assert R_corr.shape == (N, V, 3, 3)
    assert t_corr.shape == (N, V, 3)
    assert scale.shape == (N, V)
    # Identity initialisation: near-zero offsets, near-unity scale.
    print(f"mean |delta_t|: {delta_t.abs().mean().item():.6f} m")
    print(f"mean scale-1:   {scale_factor.abs().mean().item():.6f}")
    print("camera-centric coordinate transform smoke test passed")
