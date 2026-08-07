"""Anchor model augmented with a cross-view contrastive pose representation loss.

Subclasses the current iter14 anchor
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and adds a
small auxiliary contrastive head that operates on the per-joint spatio-temporal
features after principal-point correction.  The main forward/triangulation path
remains identical to the anchor; the new loss is returned alongside the normal
outputs so the training script can mix it in with a single extra term.
"""

import torch
import torch.nn as nn

from ..losses.crossview_pose_contrast import CrossViewJointContrastiveLoss
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCrossViewContrast(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Cross-view contrastive pose-representation extension of the anchor.

    Parameters
    ----------
    contrastive_dim:
        Dimensionality of the contrastive embedding projection.
    contrastive_temperature:
        Temperature for the InfoNCE contrastive loss.
    contrastive_loss_weight:
        Multiplicative weight applied to the returned contrastive loss.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
    """

    def __init__(
        self,
        *args,
        contrastive_dim: int = 64,
        contrastive_temperature: float = 0.07,
        contrastive_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.contrastive_dim = contrastive_dim
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_loss_fn = CrossViewJointContrastiveLoss(
            d=self.d,
            projection_dim=contrastive_dim,
            temperature=contrastive_temperature,
        )

    # ------------------------------------------------------------------ #
    # Helpers used to extract the per-joint spatio-temporal features that
    # the contrastive loss consumes.  This mirrors the first half of the
    # anchor forward, but is kept local so the parent class is untouched.
    # ------------------------------------------------------------------ #
    def _prepare_contrastive_features(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, x.device)

        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, V, J, _ = x.shape
        device = x.device

        # Prepare per-sample camera tensors and flatten time into batch.
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
            R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
            t = t.unsqueeze(0).expand(B * T, -1, -1)
        elif K.dim() == 4:
            K = K.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            R = R.unsqueeze(1).expand(B, T, -1, -1, -1).reshape(B * T, V, 3, 3)
            t = t.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, V, 3)
        else:
            raise ValueError("K must have shape (V, 3, 3) or (B, V, 3, 3)")

        x_flat = x.reshape(B * T, V, J, 3)
        confidences = x_flat[..., 2]

        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]

        feat = self._extract_frame_features(x_flat, K_corrected, R, t)
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # Spatio-temporal (time + view) attention, same as the anchor.
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(
            B * T, V, J, self.d
        )

        return feat  # (B*T, V, J, d)

    def compute_contrastive_loss(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the raw (unweighted) cross-view contrastive loss for ``x``."""
        feat = self._prepare_contrastive_features(x, cameras=cameras, K=K, R=R, t=t)
        return self.contrastive_loss_fn(feat)

    def forward_with_contrastive_loss(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor | None = None,
        R: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> tuple:
        """Run the anchor forward pass and also return the contrastive loss.

        This is a single-pass integration: the spatio-temporal transformer
        output is captured with a forward hook and fed to the contrastive head,
        while the anchor forward still produces ``pred_3d`` and ``weights``.
        """
        # Hook to capture the last spatio-temporal transformer layer output.
        captured = []

        def _hook(module, input, output):
            captured.append(output)

        handle = self.st_transformer[-1].register_forward_hook(_hook)
        try:
            outputs = super().forward(x, cameras=cameras, K=K, R=R, t=t)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Could not capture spatio-temporal features.")

        # Re-derive B/T/V/J from the input so we can reshape the captured tensor
        # back to (B*T, V, J, d).
        squeeze = x.dim() == 4
        if squeeze:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        B, T, V, J, _ = x_in.shape

        feat = captured[0]  # (B*J, T*V, d)
        feat = (
            feat.view(B, J, T, V, self.d)
            .permute(0, 2, 3, 1, 4)
            .reshape(B * T, V, J, self.d)
        )

        c_loss = self.contrastive_loss_fn(feat) * self.contrastive_loss_weight
        return (*outputs, c_loss)
