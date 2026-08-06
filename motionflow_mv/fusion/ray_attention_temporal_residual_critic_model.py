"""Temporal ray-aware attention fusion with residual refinement and a pose critic.

Extends ``RayAttentionFusionModelTemporalResidual`` by adding a small
self-critic temporal refiner on top of the residual-corrected 3D trajectory.
The critic looks at the residual-corrected pose sequence, conditions on the
per-joint temporal features, and predicts a further per-joint correction
which is added back to the residual output.

Input:
    x: (B, T, V, J, 3) or (B, V, J, 3) containing (x_pixel, y_pixel, confidence)
    cameras: list of Camera objects (V,)  -- single rig, broadcast over batch/time
    OR
    K: (B, V, 3, 3), R: (B, V, 3, 3), t: (B, V, 3)  -- per-sample rigs

Output:
    X: (B, T, J, 3) refined world-coordinate 3D joints, or (B, J, 3) for 4D input
    weights: (B, T, V, J) predicted per-view per-joint weights, or (B, V, J)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from .ray_attention_model import _triangulate_weighted_dlt
from ..calibration.camera import Camera


def _cameras_to_tensors(cameras: List[Camera], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


class PoseCriticTemporalRefiner(nn.Module):
    """Small self-critic network that refines a 3D pose trajectory.

    The critic embeds the current 3D pose, concatenates it with per-joint
    temporal features, and runs a lightweight temporal transformer to predict
    a per-joint correction.

    Parameters
    ----------
    j : int
        Number of joints.
    d : int
        Feature dimension used by the base model.
    n_heads : int
        Number of attention heads in the temporal critic.
    n_layers : int
        Number of temporal transformer layers.
    hidden : int
        Hidden dimension of the pose embedding MLP.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden: int = 128,
    ):
        super().__init__()
        self.j = j
        self.d = d

        # Embed the current 3D pose and fuse it with the per-joint features.
        self.pose_embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Temporal transformer operating on (B*J, T, d).
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Final correction head.
        self.corr_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, pose: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pose : (B, T, J, 3)
            Current 3D pose trajectory.
        feat : (B*T, J, d)
            Per-joint pooled temporal features from the base model.

        Returns
        -------
        delta : (B, T, J, 3)
            Correction proposed by the critic.
        """
        B, T, J, _ = pose.shape

        # Embed pose and reshape features to match.
        pose_emb = self.pose_embed(pose)  # (B, T, J, d)
        feat = feat.view(B, T, J, self.d)

        # Fuse pose embedding with temporal features.
        x = torch.cat([pose_emb, feat], dim=-1)  # (B, T, J, 2d)
        x = self.fusion(x)  # (B, T, J, d)

        # Reshape to a temporal sequence per joint.
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, self.d)  # (B*J, T, d)
        for layer in self.temporal_layers:
            x = layer(x)

        # Predict correction.
        delta = self.corr_head(x)  # (B*J, T, 3)
        delta = delta.view(B, J, T, 3).permute(0, 2, 1, 3)  # (B, T, J, 3)
        return delta


class RayAttentionFusionModelTemporalResidualCritic(RayAttentionFusionModelTemporalResidual):
    """Temporal ray-attention fusion with residual + self-critic refinement.

    The model first runs the residual temporal ray-attention model to obtain an
    initial 3D trajectory, then passes the trajectory through a small pose
    critic that predicts a further correction.  The final output is the
    residual-corrected pose plus the critic's correction.

    Parameters
    ----------
    j, d, n_views, n_heads, n_joint_layers, n_temporal_layers, max_temporal_len :
        See ``RayAttentionFusionModelTemporal``.
    residual_hidden : int
        Hidden dimension of the residual MLP.
    critic_layers : int
        Number of temporal transformer layers in the critic.
    critic_hidden : int
        Hidden dimension of the critic's pose embedding MLP.
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
        critic_layers: int = 2,
        critic_hidden: int = 128,
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
        )
        self.critic = PoseCriticTemporalRefiner(
            j=j,
            d=d,
            n_heads=n_heads,
            n_layers=critic_layers,
            hidden=critic_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        cameras: List[Camera] = None,
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

        # Reshape to temporal sequence: each token is one (view, joint) pair.
        feat = feat.view(B, T, V, J, self.d)
        feat = feat.permute(0, 2, 3, 1, 4).reshape(B * V * J, T, self.d)
        feat = feat + self.temporal_pos_embed[:T]
        for layer in self.temporal_attn:
            feat = layer(feat)
        feat = feat.view(B, V, J, T, self.d).permute(0, 3, 1, 2, 4).reshape(B * T, V, J, self.d)

        # Per-frame weight prediction and DLT triangulation.
        feat_for_weight = feat.permute(0, 2, 1, 3)  # (B*T, J, V, d)
        w_logits = self.weight_head(feat_for_weight).squeeze(-1)  # (B*T, J, V)
        weights = torch.sigmoid(w_logits).permute(0, 2, 1)  # (B*T, V, J)
        weights = weights * confidences  # (B*T, V, J)

        Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
        P = K @ Rt
        pred_3d_raw = _triangulate_weighted_dlt(points_2d, weights, P)  # (B*T, J, 3)

        # Residual refinement head.
        feat_pooled = feat.mean(dim=1)  # (B*T, J, d)
        residual_input = torch.cat([feat_pooled, pred_3d_raw], dim=-1)  # (B*T, J, d+3)
        delta = self.residual_mlp(residual_input)  # (B*T, J, 3)
        pred_3d_residual = pred_3d_raw + delta

        # Self-critic temporal refiner.
        pred_3d_residual_seq = pred_3d_residual.view(B, T, J, 3)
        critic_delta = self.critic(pred_3d_residual_seq, feat_pooled)  # (B, T, J, 3)
        pred_3d = pred_3d_residual_seq + critic_delta

        pred_3d = pred_3d.view(B, T, J, 3)
        weights = weights.view(B, T, V, J)

        if squeeze_output:
            pred_3d = pred_3d.squeeze(1)
            weights = weights.squeeze(1)

        return pred_3d, weights


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
    model = RayAttentionFusionModelTemporalResidualCritic(j=J, d=64, n_views=V)
    pred, w = model(x, cameras=cameras)
    assert pred.shape == (B, T, J, 3)
    assert w.shape == (B, T, V, J)
    loss = pred.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("temporal residual + critic model sanity check passed")
