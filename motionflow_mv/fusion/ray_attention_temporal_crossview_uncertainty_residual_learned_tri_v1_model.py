"""Spatio-temporal ray-aware attention fusion with uncertainty, residual and learned triangulation.

Combines the cross-view spatio-temporal transformer from
``RayAttentionFusionModelTemporalCrossview``, the residual refinement head from
``RayAttentionFusionModelTemporalResidual``, the uncertainty-weighted DLT from
``RayAttentionFusionModelTemporalUncertainty``, and the differentiable
Gauss-Newton triangulation head from
``RayAttentionFusionModelTemporalResidualLearnedTri``.

The resulting model is a single multi-view pose estimator that:

1. Encodes per-view 2D observations and rays with camera-conditioned embeddings.
2. Runs joint-level and view-level attention.
3. Runs spatio-temporal attention over the (time, view) grid per joint.
4. Predicts per-view log-variance (uncertainty) and converts it to DLT weights.
5. Triangulates an initial 3D estimate with weighted DLT.
6. Refines the estimate with a differentiable Gauss-Newton solver.
7. Applies a residual MLP refinement head.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    pred_3d: (B, T, J, 3) or (B, J, 3) refined world-coordinate 3D joints
    weights: (B, T, V, J) or (B, V, J) per-view per-joint DLT weights
    log_var: (B, T, V, J) or (B, V, J) predicted log variance
    nll_loss: scalar auxiliary reprojection NLL loss (0 if not computed)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_temporal_crossview_model import RayAttentionFusionModelTemporalCrossview
from .ray_attention_model import _triangulate_weighted_dlt


def _cameras_to_tensors(cameras: List, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


def _triangulate_weighted_gauss_newton(
    points_2d: torch.Tensor,
    weights: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    init_3d: torch.Tensor,
    num_iters: int = 3,
    damping: float = 1e-6,
) -> torch.Tensor:
    """Differentiable weighted Gauss-Newton triangulation.

    Refines an initial 3D estimate by minimizing the weighted reprojection
    error.
    """
    N, V, J, _ = points_2d.shape
    X = init_3d

    fx = K[:, :, 0, 0]
    s = K[:, :, 0, 1]
    cx = K[:, :, 0, 2]
    fy = K[:, :, 1, 1]
    cy = K[:, :, 1, 2]

    eye3 = torch.eye(3, device=X.device, dtype=X.dtype).view(1, 1, 3, 3)

    for _ in range(max(1, num_iters)):
        X_cam = torch.einsum("nvab,njb->nvja", R, X) + t.unsqueeze(2)
        x_c = X_cam[..., 0]
        y_c = X_cam[..., 1]
        z_c = X_cam[..., 2]

        inv_z = 1.0 / (z_c + 1e-8)
        u = (fx[:, :, None] * x_c + s[:, :, None] * y_c + cx[:, :, None] * z_c) * inv_z
        v = (fy[:, :, None] * y_c + cy[:, :, None] * z_c) * inv_z
        proj = torch.stack([u, v], dim=-1)
        r = points_2d - proj

        J_cam = torch.zeros(N, V, J, 2, 3, device=X.device, dtype=X.dtype)
        J_cam[:, :, :, 0, 0] = fx[:, :, None] * inv_z
        J_cam[:, :, :, 0, 1] = s[:, :, None] * inv_z
        J_cam[:, :, :, 0, 2] = (cx[:, :, None] - u) * inv_z
        J_cam[:, :, :, 1, 1] = fy[:, :, None] * inv_z
        J_cam[:, :, :, 1, 2] = (cy[:, :, None] - v) * inv_z

        J_world = torch.einsum("nvjab,nvbd->nvjad", J_cam, R)

        J_world = J_world.permute(0, 2, 1, 3, 4).reshape(N, J, V * 2, 3)
        r_flat = r.permute(0, 2, 1, 3).reshape(N, J, V * 2)
        w_flat = (
            weights.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, 2)
            .reshape(N, J, V * 2)
        )

        A = torch.einsum(
            "njkp,njkq->njpq",
            J_world,
            J_world * w_flat[..., None],
        )
        b = torch.einsum(
            "njkp,njk->njp",
            J_world,
            r_flat * w_flat,
        )

        A = A + damping * eye3.expand(N, J, -1, -1)

        b = b.unsqueeze(-1)
        dx = torch.linalg.solve(A, b).squeeze(-1)
        X = X + dx

    return X


class RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(RayAttentionFusionModelTemporalCrossview):
    """Spatio-temporal ray-aware fusion with uncertainty, residual and learned triangulation.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_st_layers, max_temporal_len:
        See ``RayAttentionFusionModelTemporalCrossview``.
    residual_hidden:
        Hidden dimension of the residual MLP (default 128).
    gn_iters:
        Number of Gauss-Newton iterations (default 3).
    gn_damping:
        Diagonal damping for the GN normal equations (default 1e-6).
    uncertainty_loss_weight:
        Weight for the reprojection NLL auxiliary loss.
    log_var_min, log_var_max:
        Clamp range for predicted log-variance.
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
        gn_iters: int = 3,
        gn_damping: float = 1e-6,
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
        )
        self.residual_hidden = residual_hidden
        self.gn_iters = gn_iters
        self.gn_damping = gn_damping
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Uncertainty head predicts log-variance per (view, joint).
        self.uncertainty_head = nn.Linear(d, 1)

        # Residual refinement head.
        self.residual_mlp = nn.Sequential(
            nn.Linear(d + 3, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 3),
        )

    def _reprojection_nll(
        self,
        points_2d: torch.Tensor,
        pred_3d: torch.Tensor,
        proj_matrices: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reprojection negative log-likelihood."""
        N, V, J, _ = points_2d.shape
        ones = torch.ones(N, J, 1, device=pred_3d.device, dtype=pred_3d.dtype)
        Xh = torch.cat([pred_3d, ones], dim=-1)
        p_h = torch.einsum('nvij,nkj->nvki', proj_matrices, Xh)
        z = p_h[..., 2:3].clamp(min=1e-6)
        p_proj = p_h[..., :2] / z
        err_sq = (p_proj - points_2d).pow(2).sum(dim=-1)
        nll = 0.5 * (err_sq * torch.exp(-log_var) + log_var)
        return nll.mean()

    def forward(
        self,
        x: torch.Tensor,
        cameras=None,
        K: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None,
        n_iter: int = 1,
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

        feat = self._extract_frame_features(x_flat, K, R, t)

        feat = feat.view(B, T, V, J, self.d)
        time_emb = self.time_pos_embed[:T].view(1, T, 1, 1, self.d)
        view_emb = self.view_pos_embed[:V].view(1, 1, V, 1, self.d)
        feat = feat + time_emb + view_emb

        feat = feat.permute(0, 3, 1, 2, 4).reshape(B * J, T * V, self.d)
        for layer in self.st_transformer:
            feat = layer(feat)
        feat = feat.view(B, J, T, V, self.d).permute(0, 2, 3, 1, 4).reshape(B * T, V, J, self.d)

        # Predict uncertainty per (view, joint).
        feat_for_uncertainty = feat.permute(0, 2, 1, 3)
        log_var = self.uncertainty_head(feat_for_uncertainty).squeeze(-1)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        log_var = log_var.permute(0, 2, 1)

        precision = torch.exp(-log_var)
        weights = precision * confidences
        weights = weights.clamp(min=1e-4)

        Rt = torch.cat([R, t[..., None]], dim=-1)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)

        pred_3d_gn = _triangulate_weighted_gauss_newton(
            points_2d,
            weights,
            K,
            R,
            t,
            pred_3d_raw,
            num_iters=self.gn_iters,
            damping=self.gn_damping,
        )

        feat_pooled = feat.mean(dim=1)
        pred_3d = pred_3d_gn
        for _ in range(max(1, int(n_iter))):
            residual_input = torch.cat([feat_pooled, pred_3d], dim=-1)
            delta = self.residual_mlp(residual_input)
            pred_3d = pred_3d + delta

        nll_loss = self._reprojection_nll(points_2d, pred_3d, P, log_var)
        nll_loss = self.uncertainty_loss_weight * nll_loss

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)
        log_var = log_var.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)
            log_var = log_var.squeeze(1)

        return pred_3d, weights, log_var, nll_loss


if __name__ == "__main__":
    import numpy as np
    from ..calibration.camera import Camera

    def _make_cameras(n_views: int = 4):
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

    B, T, V, J = 2, 5, 4, 17
    cameras = _make_cameras(V)
    x = torch.rand(B, T, V, J, 3)
    model = RayAttentionFusionModelTemporalCrossviewUncertaintyResidualLearnedTriV1(j=J, d=64, n_views=V)
    pred, w, log_var, nll_loss = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    assert log_var.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("advanced multi-view fusion model v1 sanity check passed")
