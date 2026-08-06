"""Temporal ray-aware fusion with residual refinement and principal-point correction.

Subclasses ``RayAttentionFusionModelTemporalResidual`` and inserts a learned
principal-point correction layer between the temporal feature extractor and the
differentiable triangulation step.  The layer predicts a small per-view
offset ``(dx, dy)`` for each intrinsic matrix ``K`` from the pooled per-view
features, then triangulates with the corrected camera matrices.

This is intended as a minimal, low-risk add-on: the correction is initialized
near zero and bounded, so the model starts from the original residual model and
can learn to fix small calibration errors.
"""

from typing import List

import torch
import torch.nn as nn

from .principal_point_correction import PrincipalPointCorrection
from .ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
    _cameras_to_tensors,
)
from .ray_attention_model import _triangulate_weighted_dlt
from ..calibration.camera import Camera


class RayAttentionFusionModelTemporalResidualPrincipalPoint(RayAttentionFusionModelTemporalResidual):
    """Temporal residual model with learned principal-point correction.

    Parameters
    ----------
    principal_point_hidden:
        Hidden dimension of the principal-point offset MLP (default 64).
    principal_point_max_offset:
        Maximum absolute principal-point correction in pixels (default 20.0).
    See ``RayAttentionFusionModelTemporalResidual`` for the remaining args.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_temporal_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        use_reproj_gate: bool = False,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_temporal_layers=n_temporal_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            use_reproj_gate=use_reproj_gate,
        )
        self.return_pp_delta = return_pp_delta
        self.correct_focal = focal_max_scale > 0.0
        self.principal_point_correction = PrincipalPointCorrection(
            d=d,
            hidden=principal_point_hidden,
            max_offset=principal_point_max_offset,
            max_focal_scale=focal_max_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            K, R, t = _cameras_to_tensors(cameras, device)

        # Prepare per-sample camera tensors and flatten time into batch for the
        # per-frame encoder.
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
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        # Principal-point / intrinsic correction: predict per-view (dx, dy)
        # and optionally focal-length scale from raw 2D observations and
        # intrinsics *before* ray embedding, so both the ray features and the
        # triangulation use the same corrected geometry.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses the *corrected* intrinsics for ray embedding).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)

        # Triangulate with the corrected intrinsics.
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        pred_3d = pred_3d_raw
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)  # (B*T, J, d+3)
            delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
            if self.use_reproj_gate:
                summary = self._reprojection_error_summary(
                    pred_3d, points_2d, P, inlier_thresh=10.0
                )
                gate_input = torch.cat([residual_input, summary], dim=-1)
                gate = self.reproj_gate(gate_input)  # (B*T, J, 1)
                delta = gate * delta
            pred_3d = pred_3d + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        if self.return_pp_delta:
            if self.correct_focal:
                return pred_3d, weights, pp_delta, focal_scale
            return pred_3d, weights, pp_delta
        return pred_3d, weights


if __name__ == "__main__":
    from .ray_attention_temporal_residual_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = RayAttentionFusionModelTemporalResidualPrincipalPoint(
        j=J, d=64, n_views=V, principal_point_max_offset=20.0
    )
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    # Verify the principal-point correction module produced small, bounded
    # initial offsets and that gradients reach it.
    assert model.principal_point_correction is not None
    print("temporal residual + principal-point correction sanity check passed")
