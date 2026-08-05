"""Cross-view spatio-temporal ray-aware fusion with adaptive view selection.

Extends ``RayAttentionFusionModelTemporalCrossviewResidual`` by inserting an
``AdaptiveViewSelector`` before triangulation.  The selector predicts a
per-view, per-joint binary mask that gates the learned fusion weights, so
occluded or unreliable views contribute less to the DLT solve.
"""

import torch

from .ray_attention_temporal_crossview_residual_model import RayAttentionFusionModelTemporalCrossviewResidual
from .adaptive_view_selector import AdaptiveViewSelector


class RayAttentionFusionModelTemporalCrossviewResidualAdaptiveViewSelection(
    RayAttentionFusionModelTemporalCrossviewResidual
):
    """Cross-view residual model with adaptive view selection.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_st_layers, max_temporal_len,
    residual_hidden:
        See ``RayAttentionFusionModelTemporalCrossviewResidual``.
    selector_k:
        Number of views to select at inference (default 4).
    selector_tau:
        Gumbel-softmax temperature during training.
    selector_geo_features:
        Whether to feed ray-geometry features to the selector.
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
        selector_k: int = 4,
        selector_tau: float = 0.5,
        selector_geo_features: bool = True,
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
        )
        self.selector = AdaptiveViewSelector(
            d=d,
            n_views=n_views,
            k=selector_k,
            tau=selector_tau,
            geo_features=selector_geo_features,
            hard_inference=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
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
            from .ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

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

        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)

        # Adaptive view selection mask.
        _, select_mask, _ = self.selector(feat, points_2d, K, R, t)  # (B*T, V, J)
        weights = weights * confidences * select_mask
        weights = weights.clamp(min=1e-4)

        from .ray_attention_model import _triangulate_weighted_dlt
        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        feat_pooled = feat.mean(dim=1)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)
        delta = self.residual_mlp(residual_input)
        pred_3d = pred_3d_raw + delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights


if __name__ == "__main__":
    from .ray_attention_temporal_crossview_model import _make_cameras

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalCrossviewResidualAdaptiveViewSelection(
        j=J, d=64, n_views=V, selector_k=2
    )
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("Cross-view adaptive view selection sanity check passed")
