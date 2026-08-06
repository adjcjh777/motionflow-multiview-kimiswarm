"""Camera-parameter-conditioned variant of the iter14 anchor model.

This model keeps the principal-point correction, cross-view spatio-temporal
attention and DLT triangulation path of
``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and only
replaces the view-weight head and the residual refinement head with variants that
explicitly consume the calibrated camera matrices (K, R, t).  The goal is to
make view selection and the final 3D residual correction aware of the physical
camera rig geometry without altering the rest of the architecture.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class _CameraConditionedWeightHead(nn.Module):
    """View-weight head that fuses per-joint features with camera parameters.

    Parameters
    ----------
    d:
        Feature dimension of the per-joint per-view tokens.
    n_views:
        Number of views (kept for API symmetry; inferred at runtime).
    cond_dim:
        Dimension of the camera-condition vector injected into each view.
    hidden:
        Hidden dimension of the final MLP.
    """

    def __init__(self, d: int, n_views: int = 4, cond_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.d = d
        self.cond_dim = cond_dim
        # Encode flattened (K, R, t) per view.
        self.cam_encoder = nn.Sequential(
            nn.Linear(9 + 9 + 3, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return view weight logits (N, J, V).

        ``feat`` has shape ``(N, J, V, d)``.
        ``K, R, t`` have shapes ``(N, V, 3, 3)`` and ``(N, V, 3)``.
        """
        N, J, V, _ = feat.shape
        # Flatten camera matrices and encode per view.
        cam_feat = torch.cat(
            [K.view(N, V, -1), R.view(N, V, -1), t.view(N, V, -1)],
            dim=-1,
        )  # (N, V, 21)
        cond = self.cam_encoder(cam_feat)  # (N, V, cond_dim)
        # Broadcast condition across joints and align with feat.
        cond = cond.unsqueeze(1).expand(N, J, V, self.cond_dim)
        x = torch.cat([feat, cond], dim=-1)  # (N, J, V, d + cond_dim)
        logits = self.mlp(x).squeeze(-1)  # (N, J, V)
        return logits


class _CameraConditionedResidualRefiner(nn.Module):
    """Residual refinement head conditioned on global camera geometry.

    Parameters
    ----------
    d:
        Dimension of the pooled per-joint feature.
    cond_dim:
        Dimension of the pooled rig-condition vector.
    hidden:
        Hidden dimension of the residual MLP.
    """

    def __init__(self, d: int, cond_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.rig_encoder = nn.Sequential(
            nn.Linear(9 + 9 + 3, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d + 3 + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        feat: torch.Tensor,
        raw3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-joint 3D residual (N, J, 3).

        ``feat``:    (N, J, d)
        ``raw3d``:   (N, J, 3)
        ``K, R, t``: (N, V, 3, 3) / (N, V, 3)
        """
        N, J, _ = feat.shape
        # Pool camera parameters across the rig, then encode to a rig condition.
        cam_feat = torch.cat(
            [K.view(N, -1, 9), R.view(N, -1, 9), t.view(N, -1, 3)],
            dim=-1,
        )  # (N, V, 21)
        rig_cond = self.rig_encoder(cam_feat.mean(dim=1))  # (N, cond_dim)
        rig_cond = rig_cond.unsqueeze(1).expand(N, J, self.cond_dim)
        x = torch.cat([feat, raw3d, rig_cond], dim=-1)
        return self.mlp(x)


class RayAttentionFusionModelTemporalCrossviewResidualCameraConditioned(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Camera-parameter-conditioned anchor model.

    Replaces the learned weight head and residual MLP with modules that consume
    the calibrated camera matrices.  All other behaviour (principal-point
    correction, spatio-temporal attention, DLT triangulation, returned tensors)
    matches the parent class.

    Parameters
    ----------
    camera_condition_dim:
        Dimension of the injected camera-condition vector (default 32).
    weight_head_hidden:
        Hidden dimension of the camera-conditioned weight head (default 128).
    residual_condition_dim:
        Dimension of the rig-condition vector for the residual refiner
        (default 32).
    See ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` for
    the remaining arguments.
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
        camera_condition_dim: int = 32,
        weight_head_hidden: int = 128,
        residual_condition_dim: int = 32,
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
        # Replace the inherited weight head and residual MLP with camera-conditioned versions.
        self.weight_head = _CameraConditionedWeightHead(
            d=d, n_views=n_views, cond_dim=camera_condition_dim, hidden=weight_head_hidden
        )
        self.residual_mlp = _CameraConditionedResidualRefiner(
            d=d, cond_dim=residual_condition_dim, hidden=residual_hidden
        )
        self.camera_condition_dim = camera_condition_dim
        self.weight_head_hidden = weight_head_hidden
        self.residual_condition_dim = residual_condition_dim

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

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight, K_corrected, R, t)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head conditioned on rig geometry.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        delta = self.residual_mlp(feat_pooled, pred_3d_raw, K_corrected, R, t)  # (B*T, J, 3)
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

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_raw:
                out.append(pred_3d_raw.view(B, T, J, 3))
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        if self.return_raw:
            return pred_3d, weights, pred_3d_raw.view(B, T, J, 3)
        return pred_3d, weights
