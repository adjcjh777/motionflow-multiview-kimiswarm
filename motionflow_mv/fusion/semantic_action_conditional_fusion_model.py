"""Semantic action-conditional multi-view fusion.

Extends the 9.32 mm anchor ``RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint``
by conditioning the spatio-temporal fusion and residual refinement on a discrete
action label.  Action information is injected through (1) an additive per-feature
embedding, (2) per-layer FiLM affine modulation of the spatio-temporal tokens,
and (3) an action-aware residual refinement head.  The change is fully backward
compatible: passing ``action_id=None`` falls back to the anchor behaviour.
"""

from typing import Optional

import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSemanticActionConditional(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Anchor + semantic action-conditional fusion (additive + FiLM + residual).

    Parameters
    ----------
    num_actions:
        Number of action categories in the dataset (e.g. 16 for Human3.6M plus a
        generic id).  The embedding table reserves one extra index for unknown.
    action_embed_dim:
        Dimensionality of the learned action embedding.  Defaults to ``d`` so it
        can be projected into the feature space with a single linear layer.
    All other arguments are forwarded to the parent PP anchor.
    """

    def __init__(
        self,
        num_actions: int = 16,
        action_embed_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim if action_embed_dim is not None else self.d

        # Action embedding table.  Index ``num_actions`` is a reserved "unknown".
        self.action_embed = nn.Embedding(num_actions + 1, self.action_embed_dim)

        # Project action embedding into the feature space for the additive path.
        self.action_to_feat = nn.Linear(self.action_embed_dim, self.d)

        # Per-layer FiLM generators for the spatio-temporal transformer.
        self.film_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.action_embed_dim, self.action_embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_dim, 2 * self.d),
                )
                for _ in self.st_transformer
            ]
        )

        # Project action embedding for the residual refinement head.
        self.action_residual_proj = nn.Linear(self.action_embed_dim, self.d)

        # Replace the inherited residual MLP with one that also consumes the
        # action embedding (concatenated, not added).
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.d + 3 + self.d, self.residual_hidden),
            nn.ReLU(),
            nn.Linear(self.residual_hidden, self.residual_hidden),
            nn.ReLU(),
            nn.Linear(self.residual_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_id: Optional[torch.Tensor] = None,
        cameras=None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ):
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if action_id is None:
            action_id = torch.full((B,), self.num_actions, dtype=torch.long, device=device)
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

        # Action embedding per batch sample.
        action_emb = self.action_embed(action_id)  # (B, action_embed_dim)

        # Spatio-temporal (time + view) attention.
        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        # Additive semantic action embedding (broadcast over T, V, J).
        action_feat = self.action_to_feat(action_emb)  # (B, d)
        feat = feat + action_feat.view(B, 1, 1, 1, self.d)

        # Prepare FiLM parameters: replicate action embedding for each joint token.
        action_emb_for_film = action_emb.unsqueeze(1).expand(B, J, -1).reshape(B * J, -1)

        # (B, J, T, V, d) -> (B*J, T*V, d)
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for i, layer in enumerate(self.st_transformer):
            film = self.film_layers[i](action_emb_for_film)  # (B*J, 2d)
            gamma, beta = film.chunk(2, dim=-1)  # each (B*J, d)
            gamma = gamma.unsqueeze(1)  # (B*J, 1, d)
            beta = beta.unsqueeze(1)  # (B*J, 1, d)
            feat = (1.0 + gamma) * feat + beta
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
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Action-conditional residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        action_emb_for_res = (
            self.action_residual_proj(action_emb)
            .unsqueeze(1)
            .expand(B, T, -1)
            .reshape(B * T, 1, self.d)
            .expand(B * T, J, self.d)
        )
        residual_input = torch.cat([feat_pooled, pred_3d_raw, action_emb_for_res], dim=-1)
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

        if self.return_pp_delta:
            out = [pred_3d, weights, pp_delta]
            if self.correct_focal:
                out.insert(3, focal_scale)
            if self.return_visibility:
                out.append(visibility)
            return tuple(out)
        if self.return_visibility:
            return pred_3d, weights, visibility
        return pred_3d, weights


if __name__ == "__main__":
    # CPU sanity check: forward + backward with random action labels.
    B, T, V, J = 2, 9, 4, 17
    d = 32
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointSemanticActionConditional(
        j=J,
        d=d,
        n_views=V,
        n_heads=2,
        n_joint_layers=1,
        n_st_layers=2,
        max_temporal_len=256,
        residual_hidden=64,
        principal_point_hidden=32,
        principal_point_max_offset=20.0,
        num_actions=16,
        action_embed_dim=d,
        return_pp_delta=True,
    )

    x = torch.randn(B, T, V, J, 3)
    K = torch.eye(3).unsqueeze(0).repeat(V, 1, 1)
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0
    R = torch.eye(3).unsqueeze(0).expand(V, -1, -1)
    t = torch.tensor([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [-1.5, 0.0, 0.0], [0.0, -1.5, 0.0]])
    action_id = torch.randint(0, 16, (B,))

    pred, weights, pp_delta = model(x, action_id=action_id, K=K, R=R, t=t)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert pp_delta.shape == (B * T, V, 2)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    print("semantic action-conditional fusion model sanity check passed")
