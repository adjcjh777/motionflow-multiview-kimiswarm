"""View-synchronised, temporally coherent multi-view keypoint augmentation.

Real multi-view setups rarely fail as independent, high-frequency noise per
joint.  Cameras drift, vibrate, or are slightly mis-calibrated, producing 2D
displacements that are (a) shared across all joints in a view and (b) smooth
across short time windows.  The :class:`MultiViewSyncAugmentation` class mimics
this structure: per-view affine jitter that is constant inside a temporal
sub-clip, plus a small independent per-joint noise term.
"""

from typing import Optional

import torch
import torch.nn as nn


class MultiViewSyncAugmentation(nn.Module):
    """Apply view-synchronised, temporally coherent jitter to 2D keypoints.

    The input is assumed to be a tensor of shape ``(B, T, V, J, C)`` where the
    last channel contains at least ``[u, v, confidence]``.  The augmentation:

    1. Splits the temporal dimension ``T`` into sub-clips of length
       ``subclip_len``.
    2. For each sub-clip, samples a per-view 2D translation, rotation and scale.
    3. Applies the sampled affine transform to **all** joints in the sub-clip
       for that view.
    4. Adds a small independent Gaussian noise per joint.
    5. Optionally drops whole views while keeping at least ``min_views``.

    All sampling is performed with ``torch.no_grad`` and the module has no
    learnable parameters.

    Parameters
    ----------
    subclip_len:
        Temporal length over which the same per-view transform is held constant.
    translation_std:
        Standard deviation (in pixels) of the per-view 2D translation.
    rotation_std_deg:
        Standard deviation (in degrees) of the per-view in-plane rotation.
    scale_std:
        Standard deviation of the per-view log-scale jitter.  A value of ``0.02``
        corresponds to roughly +/-2 % scale changes.
    noise_std:
        Standard deviation of independent per-joint Gaussian noise (pixels).
    view_dropout_rate:
        Probability of dropping an entire view during training.
    min_views:
        Minimum number of views to retain when view dropout is active.
    """

    def __init__(
        self,
        subclip_len: int = 5,
        translation_std: float = 2.0,
        rotation_std_deg: float = 1.0,
        scale_std: float = 0.02,
        noise_std: float = 0.5,
        view_dropout_rate: float = 0.0,
        min_views: int = 2,
    ):
        super().__init__()
        if subclip_len < 1:
            raise ValueError(f"subclip_len must be >= 1, got {subclip_len}")
        self.subclip_len = subclip_len
        self.translation_std = translation_std
        self.rotation_std = rotation_std_deg * 3.14159265 / 180.0
        self.scale_std = scale_std
        self.noise_std = noise_std
        self.view_dropout_rate = view_dropout_rate
        self.min_views = min_views

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to ``x``.

        Args:
            x: Tensor of shape ``(B, T, V, J, C)`` with ``C >= 3``.

        Returns:
            Augmented tensor with the same shape as ``x``.
        """
        if not self.training:
            return x

        if x.dim() < 5:
            raise ValueError(
                f"Expected input with at least 5 dimensions (B, T, V, J, C), got shape {x.shape}"
            )

        x = x.clone()
        B, T, V, J, C = x.shape
        device = x.device

        # Sub-clip boundaries.
        num_subclips = (T + self.subclip_len - 1) // self.subclip_len
        with torch.no_grad():
            for s in range(num_subclips):
                start = s * self.subclip_len
                end = min(start + self.subclip_len, T)

                # Per-sample, per-view affine parameters.
                if self.translation_std > 0.0:
                    txy = torch.randn(B, V, 2, device=device) * self.translation_std
                else:
                    txy = torch.zeros(B, V, 2, device=device)

                if self.scale_std > 0.0:
                    scale = torch.exp(torch.randn(B, V, 1, device=device) * self.scale_std).unsqueeze(1)
                else:
                    scale = torch.ones(B, 1, V, 1, device=device)

                if self.rotation_std > 0.0:
                    theta = torch.randn(B, V, 1, device=device) * self.rotation_std
                    # Add singleton dims for temporal and joint broadcasting.
                    c = torch.cos(theta).unsqueeze(1)
                    s_ = torch.sin(theta).unsqueeze(1)
                else:
                    c = torch.ones(B, 1, V, 1, device=device)
                    s_ = torch.zeros(B, 1, V, 1, device=device)

                # Apply per-view rotation + scale + translation.
                # points shape: (B, end-start, V, J, 2)
                points = x[:, start:end, :, :, :2]

                # Rotate and scale around the origin.
                u = points[..., 0]
                v = points[..., 1]
                u_rot = scale * (c * u - s_ * v)
                v_rot = scale * (s_ * u + c * v)

                # Add per-view translation (broadcasts over J).
                u_aug = u_rot + txy[:, None, :, None, 0]
                v_aug = v_rot + txy[:, None, :, None, 1]

                x[:, start:end, :, :, 0] = u_aug
                x[:, start:end, :, :, 1] = v_aug

            # Independent per-joint noise.
            if self.noise_std > 0.0:
                x[..., :2] = x[..., :2] + torch.randn_like(x[..., :2]) * self.noise_std

            # View dropout: zero the confidence channel for dropped views.
            if self.view_dropout_rate > 0.0 and V > 1:
                view_mask = (
                    torch.rand(B, V, device=device) >= self.view_dropout_rate
                ).float()  # 1 = keep
                for i in range(B):
                    kept = view_mask[i].nonzero(as_tuple=True)[0]
                    if kept.numel() < self.min_views:
                        dropped = (view_mask[i] == 0).nonzero(as_tuple=True)[0]
                        needed = self.min_views - kept.numel()
                        if needed > 0 and dropped.numel() > 0:
                            perm = torch.randperm(dropped.numel(), device=device)
                            extra = dropped[perm[:needed]]
                            view_mask[i, extra] = 1.0
                x[..., 2] = x[..., 2] * view_mask.view(B, 1, V, 1)

        return x

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"subclip_len={self.subclip_len}, "
            f"translation_std={self.translation_std}, "
            f"rotation_std_deg={self.rotation_std * 180.0 / 3.14159265:.2f}, "
            f"scale_std={self.scale_std}, "
            f"noise_std={self.noise_std}, "
            f"view_dropout_rate={self.view_dropout_rate}, "
            f"min_views={self.min_views})"
        )
