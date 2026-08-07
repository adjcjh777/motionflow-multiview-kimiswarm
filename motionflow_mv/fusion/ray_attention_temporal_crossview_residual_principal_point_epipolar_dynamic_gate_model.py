"""Anchor + epipolar bias + dynamic view-selection gate.

Combines two existing ablations into a single minimally invasive next step:

* Epipolar-line distance bias on the per-view weight head
  (geometry-aware weighting).
* A lightweight per-view/per-joint dynamic gate
  (learned occlusion/robustness).

The result is a more robust multi-view fusion model that still runs in
real time because both additions are cheap: the epipolar bias is a
closed-form geometric term and the gate is a tiny MLP.
"""

import torch
import torch.nn as nn

from .dynamic_view_selection_gate import DynamicViewSelectionGate
from .epipolar_attention_bias import (
    compute_epipolar_distance,
    epipolar_bias_from_distance,
)
from .ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)


class RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarDynamicGate(
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
):
    """Anchor with epipolar bias and dynamic view-selection gate.

    Parameters
    ----------
    epipolar_temperature:
        Temperature for converting epipolar distances to an additive weight
        bias (default 100.0 pixels).
    gate_sparsity_weight, gate_entropy_weight:
        Trainer-only regulariser weights; consumed here only to keep the
        constructor API compatible with the training script.
    return_gate:
        If True, forward returns the predicted gate weights and logits.
    """

    def __init__(
        self,
        *args,
        epipolar_temperature: float = 100.0,
        gate_sparsity_weight: float = 0.01,
        gate_entropy_weight: float = 0.001,
        return_gate: bool = True,
        **kwargs,
    ):
        # Remove trainer-only keywords so the base class never sees them.
        kwargs.pop("gate_sparsity_weight", None)
        kwargs.pop("gate_entropy_weight", None)
        super().__init__(*args, **kwargs)

        self.epipolar_temperature = epipolar_temperature
        self.return_gate = return_gate

        # Scalar blend so the model can learn to disable the epipolar bias.
        self.epipolar_gate = nn.Parameter(torch.zeros(1))

        # Per-view/per-joint soft selection gate.
        self.gate = DynamicViewSelectionGate(d=self.d, n_views=self.n_views)

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

        # Broadcast camera rig over batch and time.
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

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Optional visibility-aware weighting (base returns 1).
        visibility = self._visibility_multiplier(feat, confidences)  # (B*T, V, J)

        # Dynamic view-selection gate.
        gate_weights, gate_logits = self.gate(feat)  # (B*T, V, J)

        # Per-frame weight prediction with epipolar bias.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)

        # Epipolar bias on weight logits (geometry-aware weighting).
        epi_dist = compute_epipolar_distance(K_corrected, R, t, points_2d)
        epi_bias = epipolar_bias_from_distance(
            epi_dist, temperature=self.epipolar_temperature
        )
        epi_bias = epi_bias.permute(0, 2, 1)  # (B*T, J, V)
        blend = torch.sigmoid(self.epipolar_gate)
        w_logits = w_logits + blend * epi_bias

        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * visibility * gate_weights  # (B*T, V, J)
        weights = weights.clamp(min=1e-4)

        # Differentiable DLT triangulation.
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
        gate_weights = gate_weights.view(B, T, V, J)
        gate_logits = gate_logits.view(B, T, V, J)
        if self.return_visibility:
            visibility = visibility.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            gate_weights = gate_weights.squeeze(1)
            gate_logits = gate_logits.squeeze(1)
            if self.return_visibility:
                visibility = visibility.squeeze(1)

        raw_3d = pred_3d_raw.view(B, T, J, 3)
        if squeeze_output:
            raw_3d = raw_3d.squeeze(1)

        out = [pred_3d, weights]
        if self.return_pp_delta:
            out.append(pp_delta)
            if self.correct_focal:
                out.append(focal_scale)
        if self.return_raw:
            out.append(raw_3d)
        if self.return_visibility:
            out.append(visibility)
        if self.return_gate:
            out.extend([gate_weights, gate_logits])
        return tuple(out)


def _make_toy_cameras(n_views: int = 4, device: str = "cpu"):
    """Build a simple circular rig for the smoke test."""
    import numpy as np
    from ..calibration.camera import Camera

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


if __name__ == "__main__":
    torch.manual_seed(0)
    n_views, j, d = 4, 17, 32
    cameras = _make_toy_cameras(n_views)
    model = RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointEpipolarDynamicGate(
        j=j,
        d=d,
        n_views=n_views,
        n_st_layers=2,
        residual_hidden=64,
        principal_point_hidden=32,
        principal_point_max_offset=10.0,
        return_pp_delta=False,
        return_gate=True,
    )

    # Toy input: (B, T, V, J, 3) with confidence in the last channel.
    x = torch.randn(2, 3, n_views, j, 3)
    x[..., 2] = torch.rand_like(x[..., 2])

    pred, weights, gate_w, gate_l = model(x, cameras=cameras)
    print(f"pred shape: {pred.shape}")
    print(f"weights shape: {weights.shape}")
    print(f"gate_weights shape: {gate_w.shape}")
    print(f"gate_logits shape: {gate_l.shape}")
    assert pred.shape == (2, 3, j, 3)
    assert weights.shape == (2, 3, n_views, j)
    assert gate_w.shape == (2, 3, n_views, j)
    assert gate_l.shape == (2, 3, n_views, j)
    print("smoke test passed")
