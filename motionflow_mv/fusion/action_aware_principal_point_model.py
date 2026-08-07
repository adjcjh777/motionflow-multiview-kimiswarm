"""Action-semantics variant of the cross-view residual principal-point model.

Adds a learned action/category embedding to the spatio-temporal features of the
best-performing ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``.
The embedding is broadcast over time, views and joints and simply added to the
per-frame features before the spatio-temporal transformer, so the architecture
remains otherwise unchanged.
"""

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class ActionAwareRayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint
):
    """Principal-point model conditioned on a discrete action/category label.

    Parameters
    ----------
    num_actions:
        Number of action categories (e.g. 16 for Human3.6M).
    action_embed_dim:
        Dimension of the learned action embedding.  Defaults to ``d`` so the
        embedding can be added directly to the joint features.
    All other arguments are forwarded to the parent PP model.
    """

    def __init__(
        self,
        num_actions: int = 16,
        action_embed_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim if action_embed_dim is not None else self.d
        self.action_embed = nn.Embedding(num_actions + 1, self.action_embed_dim)

    def forward(self, x, action_id=None, cameras=None, K=None, R=None, t=None):
        # Keep the same input/output contract as the parent model but inject an
        # action embedding before the spatio-temporal transformer.
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if action_id is None:
            action_id = torch.zeros(B, dtype=torch.long, device=device)
        if action_id.dim() == 0:
            action_id = action_id.unsqueeze(0)

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

        # ---- action semantics injection --------------------------------------
        # feat currently has shape (B*T, V, J, d).  Reshape to (B, T, V, J, d),
        # add the action embedding, then flatten back.
        feat = feat.view(B, T, V, J, self.d)
        action_emb = self.action_embed(action_id)  # (B, action_embed_dim)
        # Broadcast over time, views and joints.
        action_emb = action_emb.view(B, 1, 1, 1, self.action_embed_dim).expand(B, T, V, J, self.action_embed_dim)
        feat = feat + action_emb
        feat = feat.reshape(B * T, V, J, self.d)
        # ----------------------------------------------------------------------

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction with coarse corrected intrinsics.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K_corrected @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        raw_3d = pred_3d_raw.view(B, T, J, 3)
        if squeeze_output:
            raw_3d = raw_3d.squeeze(1)

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.append(focal_scale)
            if self.return_raw:
                out.append(raw_3d)
            return tuple(out)
        if self.return_raw:
            return pred_3d, weights, raw_3d
        return pred_3d, weights
