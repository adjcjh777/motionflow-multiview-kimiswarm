"""Prototype integration of AdaptiveViewSelector into the combined model.

This script is read-only: it imports the existing combined model, monkey-patches
the selector into its forward path, and runs a synthetic forward/backward pass.
It does not train on real data.
"""

import sys
from pathlib import Path

import torch

# Load the selector from the prototype directory.
proto_dir = Path(__file__).parent
sys.path.insert(0, str(proto_dir))
sys.path.insert(0, str(proto_dir.parent.parent.parent))

from adaptive_view_selector import AdaptiveViewSelector
from motionflow_mv.fusion.ray_attention_temporal_crossview_uncertainty_residual_learned_tri_v1_model import (
    _triangulate_weighted_gauss_newton,  # noqa
    RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1,
)


def _make_dummy_rig(V=4):
    import numpy as np
    from motionflow_mv.calibration.camera import Camera

    cameras = []
    for i in range(V):
        theta = 2 * np.pi * i / V
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


def main():
    B, T, V, J, d = 2, 5, 4, 17, 64
    x = torch.rand(B, T, V, J, 3)
    cameras = _make_dummy_rig(V)

    model = RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(
        j=J, d=d, n_views=V
    )
    selector = AdaptiveViewSelector(d=d, n_views=V, k=2)

    # Monkey-patch forward to inject selection before DLT.
    base_forward = model.forward

    def adaptive_forward(x, cameras=None, K=None, R=None, t=None, n_iter=1):
        from motionflow_mv.fusion.ray_attention_model import _triangulate_weighted_dlt

        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeeze_output = True

        B, T, V, J, _ = x.shape
        device = x.device

        if K is None:
            if cameras is None:
                raise ValueError("Either cameras or (K, R, t) must be provided")
            from motionflow_mv.fusion.ray_attention_temporal_crossview_model import _cameras_to_tensors
            K, R, t = _cameras_to_tensors(cameras, device)

        K = K.unsqueeze(0).expand(B * T, -1, -1, -1)
        R = R.unsqueeze(0).expand(B * T, -1, -1, -1)
        t = t.unsqueeze(0).expand(B * T, -1, -1)

        x_flat = x.reshape(B * T, V, J, 3)
        points_2d = x_flat[..., :2]
        confidences = x_flat[..., 2]

        feat = model._extract_frame_features(x_flat, K, R, t)
        feat = feat.view(B, T, V, J, d)
        time_emb = model.time_pos_embed[:T].view(1, T, 1, 1, d)
        view_emb = model.view_pos_embed[:V].view(1, 1, V, 1, d)
        feat = feat + time_emb + view_emb
        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, d)
        for layer in model.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, d)

        log_var = model.uncertainty_head(feat.permute(0, 2, 1, 3)).squeeze(-1)
        log_var = torch.clamp(log_var, model.log_var_min, model.log_var_max)
        log_var = log_var.permute(0, 2, 1)

        # Adaptive selection mask.
        _, select_mask, _ = selector(feat, points_2d, K, R, t)
        precision = torch.exp(-log_var)
        weights = precision * confidences * select_mask
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        pred_3d_gn = _triangulate_weighted_gauss_newton(
            points_2d, weights, K, R, t, pred_3d_raw
        )
        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_gn
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = model.residual_mlp(residual_input)
            pred_3d = pred_3d + delta

        nll_loss = model._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = model.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
        return pred_3d, weights, log_var, nll_loss, select_mask

    model.forward = adaptive_forward
    pred, weights, log_var, nll_loss, mask = model(x, cameras=cameras)
    print("pred shape:", pred.shape)
    print("selected views per joint (frame 0):", mask[0, :, 0].sum(dim=0).tolist())
    loss = pred.mean() + 0.1 * nll_loss
    loss.backward()
    print("gradient check passed")


if __name__ == "__main__":
    main()
