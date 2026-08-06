"""Temporal + cross-view ray-aware attention with residual refinement,
principal-point correction, and a self-supervised masked-view 2D completion head.

Subclasses ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint`` and
adds a lightweight per-view 2D completion branch.  The branch reprojects the fused
3D skeleton into each view and refines the reprojection using the per-view
spatio-temporal feature.  During self-supervised pre-training, the refined 2D
predictions can be compared with the original (possibly masked-out) 2D
observations to encourage a physically consistent 3D reconstruction.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointCompletion(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Cross-view temporal residual model with masked-view 2D completion.

    Parameters
    ----------
    completion_hidden:
        Hidden dimension of the 2D completion MLP (default 64).
    return_completion:
        If True, the forward pass also returns the per-view completed 2D
        keypoints ``pred_2d_completed`` of shape ``(B, T, V, J, 2)``.
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
        completion_hidden: int = 64,
        return_completion: bool = False,
        **kwargs,
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
            **kwargs,
        )
        self.return_completion = return_completion

        # Completion MLP: per-view feature + reprojected 2D -> 2D residual.
        self.completion_mlp = nn.Sequential(
            nn.Linear(d + 2, completion_hidden),
            nn.ReLU(),
            nn.Linear(completion_hidden, 2),
        )

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

        # Principal-point / intrinsic correction before ray embedding and triangulation.
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
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(x_flat[..., :2], weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        # Masked-view 2D completion branch.
        pred_2d_completed = None
        if self.return_completion:
            pred_2d_completed = self._complete_2d(pred_3d, R, t, K_corrected, feat)

        # Restore shapes for standard outputs.
        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            if pred_2d_completed is not None:
                pred_2d_completed = pred_2d_completed.squeeze(1)

        # Build return tuple in the same order as the parent, with completion appended.
        out = [pred_3d, weights]
        if self.return_pp_delta:
            out.append(pp_delta)
            if self.correct_focal:
                out.append(focal_scale)
        if self.return_visibility:
            out.append(visibility.view(B, T, V, J).squeeze(1) if squeeze_output else visibility.view(B, T, V, J))
        if self.return_completion:
            out.append(pred_2d_completed)
        if self.return_raw:
            out.append(pred_3d_raw.view(B, T, J, 3))

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def _complete_2d(
        self,
        pred_3d: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        K_corrected: torch.Tensor,
        feat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Reproject fused 3D pose and refine 2D locations with per-view features.

        Args
        ----
        pred_3d: (B, T, J, 3)
        R: (B*T, V, 3, 3)
        t: (B*T, V, 3)
        K_corrected: (B*T, V, 3, 3)
        feat: (B*T, V, J, d)

        Returns
        -------
        pred_2d_completed: (B, T, V, J, 2)
        """
        B, T, J, _ = pred_3d.shape
        V = feat.shape[1]
        N = B * T

        pred_3d_flat = pred_3d.reshape(N, J, 3)

        # Camera coordinates: (N, V, J, 3)
        pred_exp = pred_3d_flat[:, None, :, :]  # (N, 1, J, 3)
        X_cam = (R[:, :, None] @ pred_exp[..., None]).squeeze(-1) + t[:, :, None, :]
        z = X_cam[..., 2:3].clamp(min=eps)  # (N, V, J, 1)
        proj = (K_corrected[:, :, None] @ X_cam[..., None]).squeeze(-1)  # (N, V, J, 3)
        proj_2d = proj[..., :2] / z  # (N, V, J, 2)

        # Completion MLP input: per-view feature + reprojected 2D.
        completion_input = torch.cat([feat, proj_2d], dim=-1)  # (N, V, J, d+2)
        pred_2d_delta = self.completion_mlp(completion_input)  # (N, V, J, 2)
        pred_2d_completed = proj_2d + pred_2d_delta
        return pred_2d_completed.view(B, T, V, J, 2)
