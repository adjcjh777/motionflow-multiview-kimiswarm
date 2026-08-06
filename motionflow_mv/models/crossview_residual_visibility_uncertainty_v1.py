"""Combined visibility-gated + uncertainty-weighted cross-view residual model.

Extends ``CrossviewResidualVisibilityV2`` by adding a per-view, per-joint
log-variance head.  The DLT weight becomes

    weight = sigmoid(weight_head) * confidence * visibility * exp(-log_var)

so occluded views are softly masked by the visibility head and noisy views are
continuously down-weighted by the predicted precision.  An auxiliary
reprojection negative-log-likelihood loss supervises the uncertainty head.
"""

import torch
import torch.nn as nn

from ..fusion.ray_attention_model import _triangulate_weighted_dlt
from .crossview_residual_visibility_v2 import CrossviewResidualVisibilityV2


class CrossviewResidualVisibilityUncertaintyV1(CrossviewResidualVisibilityV2):
    """Cross-view residual model with learned visibility and per-view uncertainty.

    Parameters
    ----------
    uncertainty_loss_weight:
        Weight for the auxiliary reprojection NLL loss (default 0.1).
    log_var_min, log_var_max:
        Clamp range for predicted log-variance.
    See ``CrossviewResidualVisibilityV2`` for the remaining arguments.
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
        visibility_hidden: int = 64,
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
        uncertainty_loss_weight: float = 0.1,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
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
            visibility_hidden=visibility_hidden,
            visibility_threshold=visibility_threshold,
            min_visible_views=min_visible_views,
        )
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Predict per-view, per-joint log-variance from the spatio-temporal features.
        self.uncertainty_head = nn.Linear(d, 1)

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood (up to constants)."""
        N, V, J, _ = points_2d.shape
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)  # (N, J, 4)
        p_h = torch.einsum("nvij,nkj->nvki", proj_matrices, Xh)  # (N, V, J, 3)
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z  # (N, V, J, 2)
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)  # (N, V, J)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

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
            from ..fusion.ray_attention_temporal_crossview_model import _cameras_to_tensors
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

        # Per-frame v3 features (uses corrected intrinsics).
        feat = self._extract_frame_features(x_flat, K_corrected, R, t)  # (B*T, V, J, d)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Visibility gating (v2 head, soft multiplier in [0, 1]).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Per-view, per-joint log-variance prediction.
        feat_for_uncertainty = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)  # (B*T, J, V)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)  # (B*T, V, J)

        # Variance-weighted DLT: lower variance -> higher precision weight.
        precision = torch.exp(-log_var)

        # Per-frame weight prediction and triangulation with corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility * precision  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        # Auxiliary reprojection NLL so uncertainties are supervised.
        nll_loss = self._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = self.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        visibility = visibility.view(B, T, V, J)
        log_var = log_var.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            visibility = visibility.squeeze(1)
            log_var = log_var.squeeze(1)

        return pred_3d, weights, visibility, log_var, nll_loss
