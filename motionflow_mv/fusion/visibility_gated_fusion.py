"""Explicit visibility-gated fusion for occluded views.

Extends ``RayAttentionFusionModelTemporalResidual`` with a small per-view,
per-joint visibility head.  The predicted soft visibility mask is applied to the
triangulation weights, so occluded or corrupted views are explicitly down-
weighted.  A fallback guard ensures that a joint is triangulated from all views
when too few views are predicted to be visible.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output (tuple):
    pred_3d: (B, T, J, 3) or (B, J, 3) world-coordinate 3D joints
    weights: (B, T, V, J) or (B, V, J) per-view per-joint DLT weights
    v_logits: (B, T, V, J) or (B, V, J) raw visibility logits
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
    _cameras_to_tensors,
)
from ..calibration.camera import Camera
from .fusion_module import FusionModule


class VisibilityGatedFusionModel(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-attention fusion with explicit per-view visibility gating.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_temporal_layers, max_temporal_len:
        See ``RayAttentionFusionModelTemporal``.
    residual_hidden:
        Hidden dimension of the residual MLP (default 128).
    use_reproj_gate:
        If True, scale the residual correction with a reprojection-error gate.
    visibility_threshold:
        Probability threshold below which a view is considered occluded.
    min_visible_views:
        Minimum number of views that must remain visible for a joint.  If fewer
        are visible, the fallback guard sets visibility to 1 for all views.
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
        visibility_threshold: float = 0.5,
        min_visible_views: int = 2,
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
        self.visibility_threshold = visibility_threshold
        self.min_visible_views = max(2, min_visible_views)

        self.visibility_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Per-frame v3 features.
        feat = self._extract_frame_features(x_flat, K, R, t)  # (B*T, V, J, d)

        # Temporal transformer: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Explicit visibility gating.
        v_logits = self.visibility_head(feat).squeeze(-1)  # (B*T, V, J)
        visibility = torch.sigmoid(v_logits)

        # Fallback guard: avoid degenerate DLT when too few views are visible.
        visible = (visibility > self.visibility_threshold).float()
        visible_count = visible.sum(dim=1)  # (B*T, J)
        fallback = (visible_count < self.min_visible_views).float().unsqueeze(1)  # (B*T, 1, J)
        effective_visibility = visibility + (1.0 - visibility) * fallback

        # Per-frame weight prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences * effective_visibility

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d_raw = self._triangulate(points_2d, weights, P, K, R, t)  # (B*T, J, 3)

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
        v_logits = v_logits.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            v_logits = v_logits.squeeze(1)

        return pred_3d, weights, v_logits


class VisibilityGatedFusionModule(FusionModule):
    """Visibility-gated fusion as a drop-in ``FusionModule`` plugin.

    Wraps ``VisibilityGatedFusionModel`` so it can be used by the rest of the
    MotionFlow multi-view pipeline.  Outputs are always in meters.
    """

    name = "visibility_gated"

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_views: int = 4,
        checkpoint_path: str | None = None,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.input_scale = input_scale
        self.model = VisibilityGatedFusionModel(j=j, d=d, n_views=n_views)
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def fuse(
        self,
        points_2d: np.ndarray,
        confidences: np.ndarray,
        cameras: List[Camera],
    ) -> np.ndarray:
        points_2d = np.asarray(points_2d, dtype=np.float32)
        confidences = np.asarray(confidences, np.float32)

        if points_2d.ndim == 3:
            points_2d = points_2d[None]
        if confidences.ndim == 2:
            confidences = confidences[None]

        # Normalise cameras to meters.
        if self.input_scale != 1.0:
            cameras = [
                Camera(
                    K=cam.K.copy(),
                    R=cam.R.copy(),
                    t=cam.t.copy() / self.input_scale,
                )
                for cam in cameras
            ]

        T, V, J, _ = points_2d.shape
        x = np.concatenate([points_2d, confidences[..., None]], axis=-1)  # (T, V, J, 3)
        x_tensor = torch.from_numpy(x).to(next(self.model.parameters()).device)

        with torch.no_grad():
            pred, *_ = self.model(x_tensor, cameras)  # (T, J, 3)
            pred = pred.cpu().numpy()
        return pred


def register_visibility_gated_fusion_module() -> None:
    """Register the visibility-gated fusion module."""
    from .fusion_module import FUSION_REGISTRY

    FUSION_REGISTRY.register(VisibilityGatedFusionModule())


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
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
    # Quick shape/gradient sanity check.
    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)

    model = VisibilityGatedFusionModel(j=J, d=64, n_views=V)
    pred, weights, v_logits = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert weights.shape == (B, T, V, J)
    assert v_logits.shape == (B, T, V, J)

    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("visibility-gated model sanity check passed")

    # Single-frame compatibility.
    x4 = torch.rand(B, V, J, 3)
    pred4, weights4, v_logits4 = model(x4, cameras=cameras)
    assert pred4.shape == (B, J, 3)
    assert weights4.shape == (B, V, J)
    assert v_logits4.shape == (B, V, J)
    print("visibility-gated single-frame sanity check passed")
