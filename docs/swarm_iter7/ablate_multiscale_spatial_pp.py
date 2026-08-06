"""CPU smoke test: SpatialFeaturePyramid inserted into the best PP model.

This script does *not* train and does *not* start any GPU work.  It only checks
that the existing ``SpatialFeaturePyramid`` block can be inserted between the
per-frame feature extractor and the spatio-temporal transformer of the best
principal-point model without breaking the forward or backward pass.

Usage
-----
    python docs/swarm_iter7/ablate_multiscale_spatial_pp.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.models.spatial_feature_pyramid import SpatialFeaturePyramid


def _make_dummy_rig(v: int = 4, batch: int = 1):
    """Return synthetic (x, K, R, t) for ``batch`` samples and ``v`` views."""
    torch.manual_seed(0)
    x = torch.rand(batch, 1, v, 17, 3)  # (B, T, V, J, 3)
    x[..., 2] = 1.0  # confidence

    K = torch.eye(3).unsqueeze(0).expand(v, -1, -1).clone()
    K[:, 0, 0] = 800.0
    K[:, 1, 1] = 800.0
    K[:, 0, 2] = 320.0
    K[:, 1, 2] = 240.0

    R = torch.eye(3).unsqueeze(0).expand(v, -1, -1).clone()
    t = torch.randn(v, 3)

    # Add batch dimension for per-sample-rig API.
    K = K.unsqueeze(0).expand(batch, -1, -1, -1)
    R = R.unsqueeze(0).expand(batch, -1, -1, -1)
    t = t.unsqueeze(0).expand(batch, -1, -1)
    return x, K, R, t


def test_pyramid_can_be_inserted():
    """Verify the pyramid consumes per-frame features and preserves gradients."""
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint(
        j=17,
        d=64,
        n_views=4,
        n_st_layers=2,
        residual_hidden=128,
    )
    model.train()

    x, K, R, t = _make_dummy_rig(v=4, batch=2)

    # 1. Extract per-frame features as the model currently does.
    feat = model._extract_frame_features(x.reshape(2 * 1, 4, 17, 3), K, R, t)
    assert feat.shape == (2, 4, 17, 64), f"Unexpected feature shape {feat.shape}"

    # 2. Reshape to (B, T, V, J, d), apply the pyramid, reshape back.
    sfp = SpatialFeaturePyramid(in_channels=64, out_channels=64, num_scales=3)
    feat_bt = feat.view(2, 1, 4, 17, 64)
    feat_pyramid = sfp(feat_bt)
    assert feat_pyramid.shape == (2, 1, 4, 17, 64)
    feat_reinserted = feat_pyramid.view(2, 4, 17, 64)

    # 3. Ensure gradients flow through the pyramid.
    loss = (feat_reinserted ** 2).mean()
    loss.backward()
    assert any(p.grad is not None for p in sfp.parameters()), "Pyramid parameters received no gradient"

    # 4. Full-model forward with the pyramid inserted (manual reinsertion).
    #    This demonstrates the exact integration point: after
    #    ``_extract_frame_features`` and before the spatio-temporal transformer.
    feat_full = model._extract_frame_features(x.reshape(2 * 1, 4, 17, 3), K, R, t)
    feat_full = feat_full.view(2, 1, 4, 17, 64)
    feat_full = sfp(feat_full)
    feat_full = feat_full.view(2 * 1, 4, 17, 64)
    assert feat_full.shape == (2, 4, 17, 64)

    print("SpatialFeaturePyramid insertion sanity test passed")
    print(f"  Input x shape:          {tuple(x.shape)}")
    print(f"  Per-frame features:     {tuple(feat.shape)}")
    print(f"  After pyramid:          {tuple(feat_full.shape)}")


def test_subclassed_integration():
    """A ready-to-use subclass sketch that wires the pyramid into the model."""

    class SpatialPyramidPPModel(RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint):
        def __init__(self, *args, num_scales: int = 3, **kwargs):
            super().__init__(*args, **kwargs)
            self.sfp = SpatialFeaturePyramid(
                in_channels=self.d,
                out_channels=self.d,
                num_scales=num_scales,
            )

        def forward(self, x, **kwargs):
            # We override forward only to insert the pyramid between feature
            # extraction and the spatio-temporal transformer.  The integration
            # is a single line (marked below).
            squeeze_output = False
            if x.dim() == 4:
                x = x.unsqueeze(1)
                squeeze_output = True

            B, T, V, J, _ = x.shape
            device = x.device

            K, R, t = kwargs.get("K"), kwargs.get("R"), kwargs.get("t")
            cameras = kwargs.get("cameras")
            if K is None and cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")

            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _cameras_to_tensors
            if K is None:
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
            feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

            # ---- Spatial feature pyramid insertion ----
            feat = feat.view(B, T, V, J, self.d)
            feat = self.sfp(feat)
            feat = feat.view(B * T, V, J, self.d)
            # -------------------------------------------

            # Continue with the existing spatio-temporal transformer.
            time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
            view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
            feat = feat.view(B, T, V, J, self.d) + time_emb + view_emb

            feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
            for layer in self.st_transformer:
                feat = layer(feat)
            feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

            points_2d = x_flat[..., :2]
            confidences = x_flat[..., 2]
            visibility = self._visibility_multiplier(feat, confidences)

            feat_for_weight = feat.permute(0, 2, 1, 3)
            w_logits = self.weight_head(feat_for_weight).squeeze(-1)
            weights = torch.sigmoid(w_logits).permute(0, 2, 1)
            weights = weights * confidences * visibility
            weights = weights.clamp(min=1e-4)

            from motionflow_mv.fusion.ray_attention_model import _triangulate_weighted_dlt
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

    model = SpatialPyramidPPModel(j=17, d=64, n_views=4, n_st_layers=2, num_scales=3)
    model.train()

    x, K, R, t = _make_dummy_rig(v=4, batch=2)
    pred, weights = model(x, K=K, R=R, t=t)
    assert pred.shape == (2, 1, 17, 3), f"Unexpected prediction shape {pred.shape}"
    assert weights.shape == (2, 1, 4, 17), f"Unexpected weight shape {weights.shape}"

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.sfp.parameters()), "Pyramid params have no gradient"

    print("Subclassed SpatialPyramidPPModel forward/backward test passed")
    print(f"  Prediction shape: {tuple(pred.shape)}")
    print(f"  Weight shape:     {tuple(weights.shape)}")


if __name__ == "__main__":
    test_pyramid_can_be_inserted()
    print()
    test_subclassed_integration()
