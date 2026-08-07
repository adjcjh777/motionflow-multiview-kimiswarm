"""Temporal + cross-view ray-aware attention with principal-point correction and
a learnable camera-centric coordinate transform.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
and adds a bounded per-view SE(3) correction before triangulation, followed by
per-view ray-depth scale fusion around the corrected camera centers.
"""

import torch
import torch.nn as nn

from .camera_centric_coordinate_transform import CameraCentricCoordinateTransform
from .ray_attention_model import _compute_rays
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


def _apply_per_view_ray_depth_scale(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Fuse per-view ray-depth residuals around the camera centers."""
    rays = _compute_rays(points_2d, K, R, t)
    camera_centers = -torch.einsum("bvij,bvj->bvi", R.transpose(-2, -1), t)
    relative = points_3d[:, None, :, :] - camera_centers[:, :, None, :]
    ray_depth = (relative * rays).sum(dim=-1)
    ray_delta = (
        (scale[:, :, None] - 1.0)[..., None]
        * ray_depth[..., None]
        * rays
    )
    weighted_delta = (ray_delta * weights[..., None]).sum(dim=1)
    return points_3d + weighted_delta / weights.sum(dim=1)[..., None]


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCameraCentric(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual + PP model with learnable camera-centric transform.

    Parameters
    ----------
    camera_centric_hidden:
        Hidden dimension of the residual SE(3)+scale predictor.
    max_rot_offset_deg:
        Maximum rotation correction in degrees.
    max_trans_offset_m:
        Maximum absolute translation correction in meters.
    max_scale_delta:
        Maximum deviation of the per-view depth scale from 1.0.
    condition_on_deep_features:
        If ``True``, condition the coordinate transform on the deep per-view
        features output by the spatio-temporal transformer; otherwise use raw
        2D+intrinsic+extrinsic descriptors.
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining args.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        n_heads: int = 4,
        n_joint_layers: int = 1,
        n_st_layers: int = 2,
        max_temporal_len: int = 256,
        residual_hidden: int = 128,
        principal_point_hidden: int = 64,
        principal_point_max_offset: float = 20.0,
        focal_max_scale: float = 0.0,
        return_pp_delta: bool = False,
        return_visibility: bool = False,
        return_raw: bool = False,
        camera_centric_hidden: int = 64,
        max_rot_offset_deg: float = 2.0,
        max_trans_offset_m: float = 0.05,
        max_scale_delta: float = 0.05,
        condition_on_deep_features: bool = True,
    ):
        super().__init__(
            j=j,
            d=d,
            n_views=n_views,
            n_heads=n_heads,
            n_joint_layers=n_joint_layers,
            n_st_layers=n_st_layers,
            max_temporal_len=max_temporal_len,
            residual_hidden=residual_hidden,
            principal_point_hidden=principal_point_hidden,
            principal_point_max_offset=principal_point_max_offset,
            focal_max_scale=focal_max_scale,
            return_pp_delta=return_pp_delta,
            return_visibility=return_visibility,
            return_raw=return_raw,
        )
        self.camera_centric_transform = CameraCentricCoordinateTransform(
            d=d,
            hidden=camera_centric_hidden,
            max_rot_offset_deg=max_rot_offset_deg,
            max_trans_offset_m=max_trans_offset_m,
            max_scale_delta=max_scale_delta,
            condition_on_deep_features=condition_on_deep_features,
        )
        self.condition_on_deep_features = condition_on_deep_features

    def forward(self, x, cameras=None, K=None, R=None, t=None):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
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

        # Principal-point / intrinsic correction before ray embedding.
        correction_outputs = self.principal_point_correction(
            K=K,
            x=x_flat,
            weights=confidences,
        )
        K_corrected = correction_outputs[0]
        pp_delta = correction_outputs[1]
        focal_scale = correction_outputs[2] if self.correct_focal else None

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # (B, J, T, V, d) -> (B*J, T*V, d)
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Optional visibility-aware weighting (base returns 1).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Deep per-view features for the camera-centric transform.
        if self.condition_on_deep_features:
            feat_pooled_per_view = (feat * confidences.unsqueeze(-1)).sum(dim=2) / (
                confidences.sum(dim=2, keepdim=True) + 1e-8
            )  # (B*T, V, d)
        else:
            feat_pooled_per_view = None

        # Predict and apply the camera-centric coordinate transform.
        R_corrected, t_corrected, scale, delta_R, delta_t, scale_factor = (
            self.camera_centric_transform(
                R=R,
                t=t,
                feat=feat_pooled_per_view,
                x=x_flat,
                K=K_corrected,
                weights=confidences,
            )
        )

        # Per-frame weight prediction and triangulation with corrected intrinsics
        # and corrected extrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R_corrected, t_corrected[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        pred_3d_raw = _apply_per_view_ray_depth_scale(
            pred_3d_raw,
            points_2d,
            scale,
            weights,
            K_corrected,
            R_corrected,
            t_corrected,
        )

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        if self.return_visibility:
            visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            if self.return_visibility:
                visibility = visibility.squeeze(1)

        raw_3d = pred_3d_raw.view(B, T, J, 3)
        if squeeze_output:
            raw_3d = raw_3d.squeeze(1)

        # Compose extra outputs (same semantics as the anchor model).
        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(raw_3d)
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        if self.return_raw:
            return pred_3d, weights, raw_3d
        return pred_3d, weights
