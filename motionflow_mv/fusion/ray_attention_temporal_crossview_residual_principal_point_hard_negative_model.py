"""Anchor model extended with failure-driven hard-negative mining.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
and adds a lightweight synthetic hard-negative generator that corrupts the most
trusted views during training.  At inference time the forward pass is identical
to the anchor, so the change is training-time only.
"""

from typing import Optional

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class SyntheticHardNegativeGenerator(nn.Module):
    """Generate hard negatives by corrupting the most confident views.

    Given a multi-view 2D keypoint tensor and camera parameters, the generator
    triangulates a raw 3D pose, ranks views by their reprojection error, and
    perturbs the best-matching (i.e. most misleading) views.  This creates
    training examples where the input looks consistent but is actually wrong,
    which are classic hard negatives for multi-view triangulation.

    Parameters
    ----------
    rot_std:
        Rotation noise std in degrees applied to easy views.
    trans_std:
        Translation noise std in the same unit as ``t``.
    pixel_outlier_std:
        Std of 2-D outlier blob added to 2D keypoints of easy views (pixels).
    corruption_ratio:
        Fraction of views to corrupt (0..1).
    """

    def __init__(
        self,
        rot_std: float = 0.3,
        trans_std: float = 0.005,
        pixel_outlier_std: float = 8.0,
        corruption_ratio: float = 0.25,
    ):
        super().__init__()
        self.rot_std = rot_std
        self.trans_std = trans_std
        self.pixel_outlier_std = pixel_outlier_std
        self.corruption_ratio = corruption_ratio

    def _triangulate_raw(
        self,
        points_2d: torch.Tensor,
        weights: torch.Tensor,
        P: torch.Tensor,
    ) -> torch.Tensor:
        """Thin wrapper around the DLT triangulation used by the anchor model."""
        from .ray_attention_model import _triangulate_weighted_dlt

        return _triangulate_weighted_dlt(points_2d, weights, P)

    def forward(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return a corrupted version of ``x`` targeting hard-negative training.

        Args:
            x:  (N, V, J, 3) tensor of (u, v, confidence).
            K:  (N, V, 3, 3) intrinsic matrices.
            R:  (N, V, 3, 3) rotation matrices.
            t:  (N, V, 3) translation vectors.

        Returns:
            x_aug: (N, V, J, 3) corrupted input tensor.
        """
        from .ray_attention_model import _triangulate_weighted_dlt

        N, V, J, _ = x.shape
        device = x.device
        points_2d = x[..., :2].clone()
        confidences = x[..., 2].clone()

        # Uniform weights for raw triangulation to obtain an initial 3D estimate.
        w = torch.ones(N, V, J, device=device) * confidences  # (N, V, J)
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (N, V, 3, 4)
        P = K @ Rt
        pred_3d = _triangulate_weighted_dlt(points_2d, w, P)  # (N, J, 3)

        # Reprojection error per (N, V).
        with torch.no_grad():
            pred_cam = torch.einsum("nvij,njk->nvik", R, pred_3d) + t.unsqueeze(2)  # (N, V, J, 3)
            proj = torch.einsum("nvij,nvjk->nvik", K, pred_cam.unsqueeze(-1)).squeeze(-1)
            z = proj[..., 2:3].clamp(min=1e-6)
            proj_2d = proj[..., :2] / z
            reproj_err = (proj_2d - points_2d).norm(dim=-1).mean(dim=-1)  # (N, V)

        x_aug = x.clone()

        # Determine how many views to corrupt per sample.
        n_corrupt = max(1, int(V * self.corruption_ratio))
        _, corrupt_indices = torch.topk(reproj_err, n_corrupt, dim=1, largest=False)  # easiest views

        # Apply translation + 2-D outlier noise to the selected easy views.
        for i in range(N):
            for v in corrupt_indices[i]:
                v = int(v.item())
                # 2-D keypoint noise.
                noise = torch.randn(J, 2, device=device) * self.pixel_outlier_std
                x_aug[i, v, :, :2] = x_aug[i, v, :, :2] + noise
                # Slightly reduce confidence to signal uncertainty.
                x_aug[i, v, :, 2] = x_aug[i, v, :, 2] * 0.5

        return x_aug


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointHardNegative(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Anchor model with a training-time hard-negative generator.

    Parameters
    ----------
    hard_negative_generator:
        Optional ``SyntheticHardNegativeGenerator`` instance.  If ``None``, a
        default generator is created.
    hard_negative_prob:
        Probability of applying the synthetic hard-negative generator to a batch
        during training.  ``0`` disables the generator (keeps the anchor
        behaviour).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    all remaining arguments.
    """

    def __init__(
        self,
        hard_negative_generator: Optional[nn.Module] = None,
        hard_negative_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hard_negative_generator = (
            hard_negative_generator if hard_negative_generator is not None else SyntheticHardNegativeGenerator()
        )
        self.hard_negative_prob = hard_negative_prob

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        """Same interface as the anchor; applies hard-negative augmentation during training.

        Args:
            x:   (B, T, V, J, 3) or (B, V, J, 3) input tensor.
            cameras, K, R, t: camera parameters (same as anchor).

        Returns:
            Same outputs as the anchor model.
        """
        if self.training and self.hard_negative_prob > 0.0:
            # Augment only when a forward hook is requested.  The generator
            # expects per-frame inputs, so we apply it after the time/view
            # flattening in the anchor forward would be too late; instead we
            # operate on the raw input and then call the parent forward.
            if torch.rand(1).item() < self.hard_negative_prob:
                # Build per-sample camera tensors for the generator.
                squeeze_output = False
                x_in = x
                if x_in.dim() == 4:
                    x_in = x_in.unsqueeze(1)
                    squeeze_output = True

                B, T, V, J, _ = x_in.shape
                device = x_in.device

                if K is None:
                    if cameras is None:
                        raise ValueError("Either cameras or (K, R, t) must be provided")
                    from .ray_attention_temporal_crossview_model import _cameras_to_tensors
                    K_p, R_p, t_p = _cameras_to_tensors(cameras, device)
                else:
                    K_p, R_p, t_p = K, R, t

                # Expand to per-frame camera tensors.
                if K_p.dim() == 3:
                    K_p = K_p.unsqueeze(0).expand(B * T, -1, -1, -1)
                    R_p = R_p.unsqueeze(0).expand(B * T, -1, -1, -1)
                    t_p = t_p.unsqueeze(0).expand(B * T, -1, -1)
                elif K_p.dim() == 4:
                    K_p = K_p.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
                    R_p = R_p.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
                    t_p = t_p.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)

                x_flat = x_in.reshape(B * T, V, J, 3)
                x_aug = self.hard_negative_generator(x_flat, K_p, R_p, t_p)
                x_in = x_aug.view(B, T, V, J, 3)
                if squeeze_output:
                    x_in = x_in.squeeze(1)
                x = x_in

        return super().forward(x, cameras=cameras, K=K, R=R, t=t)
