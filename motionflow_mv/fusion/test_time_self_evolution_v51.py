"""v51: Test-Time Self-Evolution Refiner (TTSER).

Closes the self-evolution loop at inference time.  Starting from a base pose
estimate (and optional v50 SEFH reliability / uncertainty), the module allocates
a small per-sequence buffer and refines it with a few gradient steps on a purely
geometric / self-supervised loss.  The base model weights are frozen.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestTimeSelfEvolutionRefinerV51(nn.Module):
    """Per-sequence test-time refinement of view reliability and joint uncertainty.

    Parameters
    ----------
    n_views:
        Maximum number of camera views.
    n_joints:
        Number of joints in the skeleton.
    hidden:
        Hidden dimension of the update MLP.
    num_steps:
        Number of Adam steps to run at test time.
    lr:
        Adam learning rate for the per-sequence buffer.
    reproj_weight:
        Weight of the reprojection-consistency term.
    temporal_weight:
        Weight of the temporal-smoothness term.
    bone_weight:
        Weight of the bone-length-prior term.
    entropy_weight:
        Weight of the entropy regularisation on reliability.
    min_view_rel:
        Floor for per-view reliability.
    max_view_rel:
        Ceiling for per-view reliability.
    """

    def __init__(
        self,
        n_views: int,
        n_joints: int,
        hidden: int = 32,
        num_steps: int = 3,
        lr: float = 1e-3,
        reproj_weight: float = 1.0,
        temporal_weight: float = 0.5,
        bone_weight: float = 0.1,
        entropy_weight: float = 0.01,
        min_view_rel: float = 0.05,
        max_view_rel: float = 1.0,
        refine_pose: bool = False,
        pose_lr: float = 1e-4,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.n_views = n_views
        self.n_joints = n_joints
        self.num_steps = num_steps
        self.lr = lr
        self.reproj_weight = reproj_weight
        self.temporal_weight = temporal_weight
        self.bone_weight = bone_weight
        self.entropy_weight = entropy_weight
        self.min_view_rel = min_view_rel
        self.max_view_rel = max_view_rel
        self.refine_pose = refine_pose
        self.pose_lr = pose_lr
        self.max_grad_norm = max_grad_norm

        # Update MLP: takes a small feature vector built from residuals and
        # predicts additive updates for reliability offset and log-uncertainty.
        self.update_mlp = nn.Sequential(
            nn.Linear(4 + n_views + n_joints, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_views + n_joints),
        )

    @staticmethod
    def _project(
        pose_3d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3-D pose ``(B, T, J, 3)`` to 2-D ``(B, T, V, J, 2)``.

        ``K``/``R``/``t`` are broadcast per view.  The camera convention is
        ``X_cam = R @ X + t`` and ``proj = K @ X_cam / z``.
        """
        # Rotate and translate the 3-D pose into each camera frame.
        # pose_3d: (B, T, J, 3), R: (B, V, 3, 3), t: (B, V, 3)
        X_cam = torch.einsum("bvil,btjl->btjvi", R, pose_3d) + t[:, None, None, :, :]
        # -> (B, T, J, V, 3)
        z = X_cam[..., 2:3]
        xy = X_cam[..., :2] / (z + 1e-6)  # (B, T, J, V, 2)
        # Apply intrinsics: proj = K * xy + principal point.
        # K: (B, V, 3, 3) -> use only the x2 upper block.
        xy = xy.unsqueeze(-1)  # (B, T, J, V, 2, 1)
        K_block = K[:, None, None, :, :2, :2]  # (B, 1, 1, V, 2, 2)
        proj = (K_block @ xy).squeeze(-1) + K[:, None, None, :, :2, 2]
        # -> (B, T, J, V, 2); move view axis before joint axis to match expected layout.
        return proj.permute(0, 1, 3, 2, 4)  # (B, T, V, J, 2)

    def _compute_residuals(
        self,
        pose_3d: torch.Tensor,
        x_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        bone_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a feature vector summarising geometric self-consistency.

        Args
        ----
        pose_3d: ``(B, T, J, 3)`` predicted 3-D pose.
        x_2d: ``(B, T, V, J, 2)`` 2-D keypoints with confidence in last channel if 3 channels.
        K, R, t: camera intrinsics/extrinsics, shapes ``(..., V, 3, 3)`` and ``(..., V, 3)``.
        bone_prior: ``(J-1, 3)`` optional bone-length prior.

        Returns
        -------
        ``(B, 4 + V + J)`` residual feature vector.
        """
        B, T, J, _ = pose_3d.shape
        V = x_2d.shape[-3]

        # Reprojection residual per view.
        proj = self._project(pose_3d, K, R, t)  # (B, T, V, J, 2)
        reproj = ((proj - x_2d[..., :2]) ** 2).mean(dim=(-1, -2))  # (B, T, V)
        reproj_feat = reproj.mean(dim=1)  # (B, V)

        # Temporal residual per joint.
        if T > 1:
            temp = ((pose_3d[:, 1:] - pose_3d[:, :-1]).abs()).mean(dim=(1, 3))  # (B, J)
        else:
            temp = torch.zeros(B, J, device=pose_3d.device, dtype=pose_3d.dtype)

        # Bone-length residual per joint.
        parents = torch.tensor(
            [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            device=pose_3d.device,
            dtype=torch.long,
        )
        bones = pose_3d[:, :, 1:] - pose_3d[:, :, parents[1:]]  # child - parent
        bone_len = bones.norm(dim=-1).mean(dim=1)  # (B, J - 1)
        if bone_prior is not None:
            bone_err = (bone_len - bone_prior[None, :, 0]).abs()  # naive scalar prior
            bone_err = F.pad(bone_err, (1, 0), value=0.0)  # pad root
        else:
            # If no prior, just use variance of bone lengths across time as a feature.
            bone_err = bone_len.std(dim=0, keepdim=True).expand(B, -1)
            bone_err = F.pad(bone_err, (1, 0), value=0.0)
        bone_feat = bone_err  # (B, J)

        # Scalar summaries.
        reproj_scalar = reproj_feat.mean(dim=1, keepdim=True)  # (B, 1)
        temp_scalar = temp.mean(dim=1, keepdim=True)  # (B, 1)
        bone_scalar = bone_feat.mean(dim=1, keepdim=True)  # (B, 1)
        entropy_scalar = -torch.tensor(0.0, device=pose_3d.device).expand(B, 1)

        features = torch.cat(
            [reproj_scalar, temp_scalar, bone_scalar, entropy_scalar, reproj_feat, temp],
            dim=-1,
        )
        return features  # (B, 4 + V + J)

    def _loss(
        self,
        pose_3d: torch.Tensor,
        rho: torch.Tensor,
        log_sigma: torch.Tensor,
        x_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised test-time loss.

        Args
        ----
        pose_3d: ``(B, T, J, 3)``.
        rho: ``(B, V)`` raw reliability offsets before sigmoid.
        log_sigma: ``(B, J)`` log-uncertainty before exp.
        x_2d: ``(B, T, V, J, 2 or 3)`` 2-D keypoints.
        K, R, t: cameras.

        Returns
        -------
        Scalar loss.
        """
        rel = torch.sigmoid(rho).clamp(self.min_view_rel, self.max_view_rel)
        # Clamp log_sigma to keep sigma numerically stable.
        log_sigma_clamped = log_sigma.clamp(-5.0, 5.0)
        sigma = torch.exp(log_sigma_clamped)
        B, T, J, _ = pose_3d.shape
        V = x_2d.shape[-3]

        # Reprojection loss weighted by reliability.
        proj = self._project(pose_3d, K, R, t)  # (B, T, V, J, 2)
        reproj = ((proj - x_2d[..., :2]) ** 2).mean(dim=-1)  # (B, T, V, J)
        reproj_loss = (rel[:, None, :, None] * reproj).mean()

        # Temporal loss down-weighted by uncertainty.
        if T > 1:
            temp = ((pose_3d[:, 1:] - pose_3d[:, :-1]) ** 2).mean(dim=-1)  # (B, T-1, J)
            temporal_loss = (temp / (sigma[:, None, :] + 1e-6)).mean()
        else:
            temporal_loss = torch.tensor(0.0, device=pose_3d.device)

        # Bone-length prior (encourage fixed bone lengths).
        parents = torch.tensor(
            [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            device=pose_3d.device,
            dtype=torch.long,
        )
        bones = pose_3d[:, :, 1:] - pose_3d[:, :, parents[1:]]
        bone_len = bones.norm(dim=-1)  # (B, T, J-1)
        if T > 1:
            bone_loss = bone_len.std(dim=1).mean()  # encourage low variance across time
        else:
            bone_loss = torch.tensor(0.0, device=pose_3d.device)

        # Entropy regularisation on reliability.
        entropy_loss = -torch.mean(rel * torch.log(rel + 1e-8) + (1 - rel) * torch.log(1 - rel + 1e-8))

        total = (
            self.reproj_weight * reproj_loss
            + self.temporal_weight * temporal_loss
            + self.bone_weight * bone_loss
            + self.entropy_weight * entropy_loss
        )
        return total

    def forward(
        self,
        pose_3d: torch.Tensor,
        x_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        sefh_reliability: Optional[torch.Tensor] = None,
        sefh_log_var: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine per-view reliability and per-joint uncertainty at test time.

        Args
        ----
        pose_3d: ``(B, T, J, 3)`` base 3-D pose estimate.
        x_2d: ``(B, T, V, J, C)`` 2-D keypoints (C=2 or 3 with confidence).
        K: ``(B, V, 3, 3)`` intrinsics (or ``(V, 3, 3)`` broadcasted).
        R: ``(B, V, 3, 3)`` rotations.
        t: ``(B, V, 3)`` translations.
        sefh_reliability: ``(B, V)`` optional v50 SEFH reliability seed.
        sefh_log_var: ``(B, J)`` optional v50 SEFH log-variance seed.

        Returns
        -------
        refined_pose: same shape as ``pose_3d`` (identity if ``num_steps=0``).
        refined_reliability: ``(B, V)`` after sigmoid.
        refined_uncertainty: ``(B, J)`` after exp.
        """
        B = pose_3d.shape[0]
        V = x_2d.shape[-3]
        J = pose_3d.shape[-2]
        device = pose_3d.device

        # Initialise buffer.  With no v50 seed, start at neutral values.
        if sefh_reliability is not None:
            rho = torch.logit(sefh_reliability.clamp(1e-6, 1 - 1e-6))
        else:
            rho = torch.zeros(B, V, device=device)

        if sefh_log_var is not None:
            log_sigma = sefh_log_var
        else:
            log_sigma = torch.zeros(B, J, device=device)

        # Keep rho and log_sigma as leaf tensors so autograd works.
        rho = nn.Parameter(rho.clone())
        log_sigma = nn.Parameter(log_sigma.clone())

        # Optionally also refine the pose itself at test time.  This is disabled
        # by default to preserve the previous behaviour; the model wires it on.
        if self.refine_pose:
            refined_pose = nn.Parameter(pose_3d.clone().detach())
        else:
            refined_pose = pose_3d

        # Short-circuit if no test-time steps.
        if self.num_steps <= 0:
            return refined_pose.detach(), torch.sigmoid(rho).detach(), torch.exp(log_sigma).detach()

        # Optional: an MLP-predicted update direction.  For simplicity in the
        # minimal implementation, we run gradient descent directly on the
        # geometric loss w.r.t. the buffer; the MLP is retained for future use.
        params = [rho, log_sigma]
        if self.refine_pose:
            params.append(refined_pose)
        optimizer = torch.optim.Adam(params, lr=self.lr if not self.refine_pose else self.pose_lr)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            loss = self._loss(refined_pose, rho, log_sigma, x_2d, K, R, t)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            optimizer.step()

        refined_reliability = torch.sigmoid(rho).detach().clamp(self.min_view_rel, self.max_view_rel)
        refined_uncertainty = torch.exp(log_sigma.clamp(-5.0, 5.0)).detach()

        return refined_pose.detach(), refined_reliability, refined_uncertainty
