"""Mixed-dataset temporal ray-attention fusion with residual refinement and principal-point correction.

Extends ``RayAttentionFusionModelTemporalMixedResidual`` by adding a learned
principal-point correction layer before the shared temporal backbone and
per-dataset triangulation.  The correction is applied to the (possibly padded)
intrinsic matrices before ray embedding and projection-matrix construction.
"""

import torch
import torch.nn as nn

from .principal_point_correction import PrincipalPointCorrection
from .ray_attention_temporal_mixed_residual_v1 import RayAttentionFusionModelTemporalMixedResidual


class RayAttentionFusionModelTemporalMixedResidualPrincipalPoint(RayAttentionFusionModelTemporalMixedResidual):
    """Mixed-dataset residual model with learned principal-point correction.

    Parameters
    ----------
    principal_point_hidden:
        Hidden dimension of the principal-point offset MLP (default 64).
    principal_point_max_offset:
        Maximum absolute principal-point correction in pixels (default 20.0).
    See ``RayAttentionFusionModelTemporalMixedResidual`` for the remaining args.
    """

    def __init__(
        self,
        d: int = 64,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
    ):
        # Store before super().__init__ because the parent reads _DATASET_SPECS.
        self.residual_hidden = residual_hidden
        self.return_pp_delta = return_pp_delta
        self.principal_point_hidden = principal_point_hidden
        self.principal_point_max_offset = principal_point_max_offset
        self.focal_max_scale = focal_max_scale
        self.correct_focal = focal_max_scale > 0.0

        super().__init__(
            d=d,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
        )

        self.principal_point_correction = PrincipalPointCorrection(
            d=d,
            hidden=principal_point_hidden,
            max_offset=principal_point_max_offset,
            max_focal_scale=focal_max_scale,
        )

    def _temporal_features(self, x, K, R, t):
        """Run shared per-frame + temporal encoder, with corrected intrinsics."""
        B, T, V, J, _ = x.shape
        assert (V, J) == (self.max_views, self.max_joints), "Input must be padded to max dims"

        # Flatten time for correction and per-frame encoder.
        x_flat = x.reshape(B * T, V, J, 3)
        K = K.unsqueeze(1).expand(B, T, V, 3, 3).reshape(B * T, V, 3, 3)
        R = R.unsqueeze(1).expand(B, T, V, 3, 3).reshape(B * T, V, 3, 3)
        t = t.unsqueeze(1).expand(B, T, V, 3).reshape(B * T, V, 3)

        # Intrinsic correction before ray embedding.
        confidences = x_flat[..., 2]
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Use corrected intrinsics for ray features.
        feat = self.backbone._extract_frame_features(x_flat, K_corrected, R, t)

        # Temporal transformer.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.backbone.temporal_pos_embed[:T]
        for layer in self.backbone.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        if self.correct_focal:
            return feat, K_corrected, pp_delta, focal_scale
        return feat, K_corrected, pp_delta

    def forward(self, x, K, R, t, dataset_ids):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, V, J, _ = x.shape
        assert (V, J) == (self.max_views, self.max_joints)
        device = x.device

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Shared temporal features (uses corrected intrinsics).
        temporal_outputs = self._temporal_features(x, K, R, t)
        feat = temporal_outputs[0]
        K_corrected = temporal_outputs[1]
        pp_delta = temporal_outputs[2]
        focal_scale = temporal_outputs[3] if self.correct_focal else None

        # Projection matrices from corrected intrinsics (expand over time).
        R_flat = R.unsqueeze(1).expand(B, T, V, 3, 3).reshape(B * T, V, 3, 3)
        t_flat = t.unsqueeze(1).expand(B, T, V, 3).reshape(B * T, V, 3)
        Rt = torch.cat([R_flat, t_flat[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt  # (B*T, V, 3, 4)

        pred_all = torch.zeros(B * T, self.max_joints, 3, device=device, dtype=x.dtype)
        mask_all = torch.zeros(B * T, self.max_joints, device=device, dtype=torch.bool)

        dataset_ids = dataset_ids.to(device)

        for did, spec in self._DATASET_SPECS.items():
            idx = (dataset_ids == did).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            pos = (idx[:, None] * T + torch.arange(T, device=device)[None, :]).view(-1)
            pred_branch, mask_branch = self._run_branch(
                feat[pos], points_2d[pos], confidences[pos], P[pos], spec["name"]
            )
            pred_all[pos] = pred_branch
            mask_all[pos] = mask_branch

        pred_all = pred_all.view(B, T, self.max_joints, 3)
        mask_all = mask_all.view(B, T, self.max_joints)

        if self.return_pp_delta:
            if self.correct_focal:
                return pred_all, mask_all, pp_delta, focal_scale
            return pred_all, mask_all, pp_delta
        return pred_all, mask_all
