"""Kinematic Anthropometric Prior (KAP) for OmniMultiViewFusion v22.

A lightweight, SMPL-free head that refines a 3D pose estimate using a learned
bone-length prior and an optional soft joint-angle limit.  It is designed to
stack after the residual/diffusion refinement and after v21 neural bundle
adjustment, but before the final v19 temporal Perceiver.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from motionflow_mv.fusion.graph_joint_relation import (
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)
from motionflow_mv.losses.kinematic_v15 import joint_limit_loss


# Number of bones for each supported skeleton.
_J_TO_N_BONES = {17: len([p for p in H36M_17_PARENTS if p != -1]), 28: len([p for p in MPI_INF_3DHP_28_PARENTS if p != -1])}


def _get_parents(j: int) -> List[int]:
    """Return the parent list for the given number of joints."""
    if j == 17:
        return H36M_17_PARENTS
    if j == 28:
        return MPI_INF_3DHP_28_PARENTS
    raise ValueError(f"Unsupported joint count {j}; supported: 17, 28")


class KinematicAnthropometricPrior(nn.Module):
    """Learned bone-length prior + optional joint-angle limit for 3D pose refinement.

    Parameters
    ----------
    j:
        Number of joints (17 for H36M, 28 for MPI-INF-3DHP).
    d:
        Per-joint feature dimension.
    hidden:
        Hidden dimension for the small MLPs.
    residual_hidden:
        Hidden dimension for the residual refinement MLP.
    use_angle_limit:
        Whether to add the soft joint-angle limit penalty.
    max_flexion_deg:
        Maximum allowed interior joint angle in degrees.
    max_delta:
        Maximum magnitude of the learned per-joint residual correction in meters.
    """

    def __init__(
        self,
        j: int = 17,
        d: int = 64,
        hidden: int = 64,
        residual_hidden: int = 128,
        use_angle_limit: bool = True,
        max_flexion_deg: float = 160.0,
        max_delta: float = 0.10,
    ):
        super().__init__()
        self.j = j
        self.d = d
        self.use_angle_limit = use_angle_limit
        self.max_flexion_deg = max_flexion_deg
        self.max_delta = max_delta

        parents = _get_parents(j)
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))
        # Build list of (child, parent) bones where parent != -1.
        bones = [(c, p) for c, p in enumerate(parents) if p != -1]
        self.n_bones = len(bones)
        if self.n_bones == 0:
            raise ValueError("Skeleton has no bones; cannot build KAP.")
        bone_pairs = torch.tensor(bones, dtype=torch.long)  # (n_bones, 2)
        self.register_buffer("bone_pairs", bone_pairs)

        # Per-bone mean length and log variance.
        self.bone_mu = nn.Parameter(torch.full((self.n_bones,), 0.25))
        self.bone_logvar = nn.Parameter(torch.full((self.n_bones,), math.log(0.05 ** 2)))

        # Small MLP to convert per-bone NLL into per-joint kinematic features.
        self.kinematic_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Residual refinement branch: predicts per-joint 3-D delta and 1-D confidence.
        in_dim = d + 3 + hidden
        self.residual_mlp = nn.Sequential(
            nn.Linear(in_dim, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(residual_hidden, 3)
        self.conf_head = nn.Linear(residual_hidden, 1)

        # Initialize residual branch as near-identity and confidence near 1.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.constant_(self.conf_head.bias, 2.0)

    def _bone_lengths(self, pred_3d: torch.Tensor) -> torch.Tensor:
        """Return bone lengths for each bone, shape (N, n_bones)."""
        child_joints = pred_3d[:, self.bone_pairs[:, 0], :]  # (N, n_bones, 3)
        parent_joints = pred_3d[:, self.bone_pairs[:, 1], :]  # (N, n_bones, 3)
        bone_vec = child_joints - parent_joints  # (N, n_bones, 3)
        lengths = bone_vec.norm(dim=-1)  # (N, n_bones)
        return lengths

    def _scatter_bone_to_joint(self, bone_vals: torch.Tensor) -> torch.Tensor:
        """Scatter per-bone values to per-joint features.

        Args:
            bone_vals: (N, n_bones) tensor.
        Returns:
            (N, j, 1) tensor where each joint receives the average of its
            incident bone values.
        """
        N = bone_vals.shape[0]
        out = torch.zeros(N, self.j, 1, device=bone_vals.device, dtype=bone_vals.dtype)
        # Accumulate values for child and parent joints.
        for idx, (child, parent) in enumerate(self.bone_pairs.tolist()):
            out[:, child, 0] = out[:, child, 0] + bone_vals[:, idx]
            out[:, parent, 0] = out[:, parent, 0] + bone_vals[:, idx]
        # Normalize by incident bone count per joint.
        counts = torch.zeros(self.j, device=bone_vals.device, dtype=bone_vals.dtype)
        for child, parent in self.bone_pairs.tolist():
            counts[child] += 1.0
            counts[parent] += 1.0
        out = out / counts.view(1, self.j, 1).clamp(min=1.0)
        return out

    def forward(
        self,
        feat_pooled: torch.Tensor,
        pred_3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine a 3D pose using the learned anthropometric prior.

        Args:
            feat_pooled: (N, J, d) per-joint pooled features.
            pred_3d: (N, J, 3) current 3D pose estimate.

        Returns:
            pred_3d_refined: (N, J, 3) refined pose.
            kap_loss: scalar loss tensor.
        """
        if pred_3d.dim() != 3 or pred_3d.shape[-1] != 3:
            raise ValueError(f"pred_3d must be (N, {self.j}, 3), got {pred_3d.shape}")

        # Bone-length negative log-likelihood.
        lengths = self._bone_lengths(pred_3d)  # (N, n_bones)
        mu = self.bone_mu.view(1, self.n_bones)
        logvar = self.bone_logvar.view(1, self.n_bones)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * ((lengths - mu) ** 2 * inv_var + logvar)
        bone_nll = nll.mean()

        # Build per-joint kinematic features from per-bone NLL.
        per_bone_nll = nll.detach()  # (N, n_bones)
        per_joint_nll = self._scatter_bone_to_joint(per_bone_nll)  # (N, j, 1)
        kin_feat = self.kinematic_mlp(per_joint_nll)  # (N, j, hidden)

        # Residual refinement branch.
        mlp_in = torch.cat([feat_pooled, pred_3d, kin_feat], dim=-1)  # (N, j, d+3+hidden)
        mlp_out = self.residual_mlp(mlp_in)
        delta_raw = self.delta_head(mlp_out)  # (N, j, 3)
        conf = torch.sigmoid(self.conf_head(mlp_out))  # (N, j, 1)

        delta = torch.tanh(delta_raw) * self.max_delta
        pred_3d_refined = pred_3d + conf * delta

        # Optional soft joint-angle limit penalty.
        angle_loss = torch.tensor(0.0, device=pred_3d.device)
        if self.use_angle_limit:
            parents = _get_parents(self.j)
            angle_loss = joint_limit_loss(
                pred_3d_refined,
                parents,
                max_flexion_deg=self.max_flexion_deg,
            )

        kap_loss = bone_nll + angle_loss

        return pred_3d_refined, kap_loss


if __name__ == "__main__":
    # CPU smoke test.
    B, T, J, d = 2, 9, 17, 64
    model = KinematicAnthropometricPrior(j=J, d=d)
    feat = torch.randn(B * T, J, d)
    pred = torch.randn(B * T, J, 3)
    pred_ref, loss = model(feat, pred)
    assert pred_ref.shape == (B * T, J, 3)
    assert loss.shape == ()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("KinematicAnthropometricPrior CPU smoke test passed")
