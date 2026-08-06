"""Multi-scale spatial feature pyramid over the joint dimension.

The pyramid downsamples the joint-resolution feature map by powers of two,
applies light 1-D convolutions at each scale, and upsamples the results back to
the original joint count.  Outputs are fused channel-wise so the final
representation retains both fine-grained and coarse skeleton structure.

This module is intentionally simple: it expects a feature map of shape
``(B, T, V, J, C)`` and returns a feature map of the same shape
``(B, T, V, J, C_out)``.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeaturePyramid(nn.Module):
    """Spatial feature pyramid over joints.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    out_channels:
        Number of output feature channels.
    num_scales:
        Number of pyramid scales (including the original resolution branch).
    hidden_channels:
        Intermediate channel count used inside each scale branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int = 3,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales must be at least 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.hidden_channels = hidden_channels or out_channels

        self.raw_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.down_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        for _ in range(num_scales):
            self.down_convs.append(
                nn.Conv1d(
                    in_channels,
                    self.hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            self.out_convs.append(
                nn.Conv1d(
                    self.hidden_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )

    def _target_length(self, j: int, scale: int) -> int:
        return max(1, j // (2 ** scale))

    def forward(
        self,
        x: torch.Tensor,
        return_shapes: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, T, V, J, C_in)``.
        return_shapes:
            If ``True``, return a tuple ``(output, shape_list)`` where
            ``shape_list`` records the intermediate feature map shapes at each
            pyramid scale.

        Returns
        -------
        Either ``(B, T, V, J, C_out)`` or a tuple with the shapes list.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5-D input (B,T,V,J,C), got {x.shape}")

        b, t, v, j, _ = x.shape
        # Collapse batch/time/view into one dimension; operate on joint axis.
        x_1d = x.permute(0, 1, 2, 4, 3).reshape(b * t * v, self.in_channels, j)

        out = self.raw_proj(x_1d)  # (N, C_out, J)
        shapes: List[Tuple[int, int, int]] = []

        for scale in range(self.num_scales):
            target = self._target_length(j, scale)
            pooled = F.adaptive_avg_pool1d(x_1d, target)  # (N, C_in, target)
            y = F.relu(self.down_convs[scale](pooled))  # (N, C_hid, target)
            y = self.out_convs[scale](y)  # (N, C_out, target)
            shapes.append(tuple(y.shape))

            if y.size(-1) != j:
                y = F.interpolate(
                    y,
                    size=j,
                    mode="linear",
                    align_corners=False,
                )  # (N, C_out, J)

            out = out + y

        # Restore original shape.
        out = out.view(b, t, v, self.out_channels, j).permute(0, 1, 2, 4, 3)

        if return_shapes:
            return out, shapes
        return out


class SpatialFeaturePyramidModel(nn.Module):
    """Minimal smoke-test model built around ``SpatialFeaturePyramid``.

    Consumes multi-view 2D keypoint clips ``(B, T, V, J, 3)`` and predicts
    3D joint positions ``(B, T, J, 3)``.  Camera parameters are accepted for
    API compatibility but are not used by this skeleton model.
    """

    def __init__(
        self,
        j: int = 28,
        d: int = 64,
        n_views: int = 14,
        num_scales: int = 3,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.n_views = n_views
        self.num_scales = num_scales

        self.obs_embed = nn.Linear(3, d)
        self.sfp = SpatialFeaturePyramid(
            in_channels=d,
            out_channels=d,
            num_scales=num_scales,
        )
        self.fusion = nn.Linear(d, d)
        self.pose_head = nn.Linear(d, 3)

    def forward(
        self,
        x: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        # x: (B, T, V, J, 3)
        z = self.obs_embed(x)  # (B, T, V, J, d)
        z = self.sfp(z)  # (B, T, V, J, d)

        # Aggregate across views.
        z = z.mean(dim=2)  # (B, T, J, d)
        z = F.relu(self.fusion(z))
        pred = self.pose_head(z)  # (B, T, J, 3)

        if squeeze_output:
            pred = pred.squeeze(1)
        return pred
