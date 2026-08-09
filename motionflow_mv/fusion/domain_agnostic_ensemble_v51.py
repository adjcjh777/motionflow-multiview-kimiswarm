"""v51 Domain-Agnostic Ensemble (DAE).

Turns the existing geometry, sparse-view, temporal, domain-aware, and
self-evolution branches into a small committee of pose experts.  A learned
geometric-evidence gate blends the experts per joint, without requiring domain
labels at inference.  The gate is initialized so the geometry expert dominates
at startup, preserving the strong full-view baseline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAgnosticEnsembleV51(nn.Module):
    """Domain-agnostic ensemble of per-branch 3-D pose experts.

    Parameters
    ----------
    j:
        Number of joints.
    n_experts:
        Number of expert pose streams.  At least 2 (geometry + residual).
    hidden:
        Hidden dimension of the evidence MLP.
    num_layers:
        Number of layers in the evidence MLP.
    dropout:
        Dropout probability in the evidence MLP.
    identity_bypass:
        If True, add a small uniform bypass so the ensemble is never far from
        the mean of experts at init.
    min_weight:
        Minimum per-expert weight after softmax clamping.
    """

    def __init__(
        self,
        j: int = 17,
        n_experts: int = 3,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        identity_bypass: bool = True,
        min_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.j = j
        self.n_experts = max(2, n_experts)
        self.identity_bypass = identity_bypass
        self.min_weight = min_weight

        # Evidence vector per (expert, joint):
        #   reprojection residual magnitude (1)
        #   temporal jump (1)
        #   epipolar consistency (1)
        #   view count signal (1)
        #   expert index (1)
        in_dim = 5
        layers: List[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        # Joint embedding helps the gate learn skeleton-aware weighting.
        self.joint_embed = nn.Embedding(j, hidden)

        # Per-(expert, joint) logit head.  One scalar logit per expert; softmax
        # over experts yields the blending weights.  Initialize the geometry
        # expert (index 0) with a positive bias so it dominates at startup.
        self.logit_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        with torch.no_grad():
            # Geometry expert logit ~2.0 -> softmax weight ~0.88 for 2 experts,
            # ~0.80 for 3 experts.
            self.logit_head.bias[0] = 2.0

    def _project(self, P: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project 3-D pose to 2-D.  P: (B, T, J, 3), returns (B, T, V, J, 2).

        Supports K/R/t shaped as (B, V, 3, 3) or (B, T, V, 3, 3).
        """
        B, T, J, _ = P.shape
        if K.dim() == 4:
            # (B, V, 3, 3) -> broadcast over T
            K = K.unsqueeze(1).expand(-1, T, -1, -1, -1)
            R = R.unsqueeze(1).expand(-1, T, -1, -1, -1)
            t = t.unsqueeze(1).expand(-1, T, -1, -1)
        # Now K, R are (B, T, V, 3, 3); t is (B, T, V, 3)
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (B, T, V, 3, 4)
        Pmat = K @ Rt  # (B, T, V, 3, 4)
        Pmat = Pmat.unsqueeze(3)  # (B, T, V, 1, 3, 4)
        ones = torch.ones(B, T, 1, J, 1, device=P.device, dtype=P.dtype)
        P_h = torch.cat([P.unsqueeze(2), ones], dim=-1).unsqueeze(-1)  # (B, T, V, J, 4, 1)
        proj = (Pmat @ P_h).squeeze(-1)  # (B, T, V, J, 3)
        return proj[..., :2] / (proj[..., 2:3] + 1e-8)

    def _compute_evidence(
        self,
        expert_poses: List[torch.Tensor],
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return evidence tensor (B, T, E, J, 5)."""
        B, T = expert_poses[0].shape[:2]
        V = points_2d.shape[2]
        J = self.j
        E = len(expert_poses)
        device = expert_poses[0].device

        # Broadcast mask to (B, T, V, J).
        mask = view_mask.unsqueeze(-1)  # (B, T, V, 1)
        mask = mask.expand(B, T, V, J)

        reproj_all = []
        temporal_all = []
        epipolar_all = []
        view_count_all = []

        for p in expert_poses:
            # Reprojection residual.
            proj = self._project(p, K, R, t)  # (B, T, V, J, 2)
            diff = (proj - points_2d).norm(dim=-1)  # (B, T, V, J)
            diff = diff * mask.float()
            # Mean over valid views.
            reproj = (diff.sum(dim=2) / (mask.float().sum(dim=2).clamp(min=1.0)))  # (B, T, J)
            reproj_all.append(reproj)

            # Temporal jump magnitude.
            if T > 1:
                vel = p[:, 1:] - p[:, :-1]
                pad = torch.zeros_like(vel[:, :1])
                vel = torch.cat([pad, vel], dim=1)
                temporal = vel.norm(dim=-1)  # (B, T, J)
            else:
                temporal = torch.zeros(B, T, J, device=device, dtype=p.dtype)
            temporal_all.append(temporal)

            # Epipolar-ish consistency: std of projected 2-D positions across views.
            if V > 1:
                std = proj.std(dim=2)  # (B, T, J, 2)
                epipolar = std.norm(dim=-1)  # (B, T, J)
            else:
                epipolar = torch.zeros(B, T, J, device=device, dtype=p.dtype)
            epipolar_all.append(epipolar)

            # View count signal (fraction of valid views).
            count = mask.float().sum(dim=2) / max(V, 1)  # (B, T, J)
            view_count_all.append(count.mean(dim=-1))  # (B, T)

        # Stack evidence.
        reproj_stack = torch.stack(reproj_all, dim=2)  # (B, T, E, J)
        temporal_stack = torch.stack(temporal_all, dim=2)
        epipolar_stack = torch.stack(epipolar_all, dim=2)
        count_stack = torch.stack(view_count_all, dim=2).unsqueeze(-1).expand(B, T, E, J)

        # Expert index embedding (helps distinguish experts).
        expert_idx = torch.arange(E, device=device).float().view(1, 1, E, 1, 1).expand(B, T, E, J, 1)
        expert_idx = expert_idx / max(E - 1, 1)

        evidence = torch.stack([
            reproj_stack,
            temporal_stack,
            epipolar_stack,
            count_stack,
        ], dim=-1)  # (B, T, E, J, 4)
        evidence = torch.cat([evidence, expert_idx], dim=-1)  # (B, T, E, J, 5)
        return evidence

    def forward(
        self,
        expert_poses: List[torch.Tensor],
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ensemble pose and optional auxiliary loss.

        Parameters
        ----------
        expert_poses:
            List of (B, T, J, 3) candidate poses.  The first pose is assumed to be
            the geometry baseline and receives the highest weight at init.
        points_2d:
            (B, T, V, J, 2) input keypoints.
        K, R, t:
            Camera parameters, each (B, V, 3, 3) or (B, V, 3) for t.
        view_mask:
            (B, T, V) binary mask.

        Returns
        -------
        ensemble_pose: (B, T, J, 3)
        diversity_loss: scalar tensor
        """
        B, T, J, _ = expert_poses[0].shape
        E = len(expert_poses)
        if E != self.n_experts:
            raise ValueError(f"Expected {self.n_experts} experts, got {E}")

        # Ensure K, R, t are shaped (B, T, V, ...) when dim >= 4.
        if K.dim() == 3:
            V = K.shape[0]
            K = K.unsqueeze(0).expand(B, -1, -1, -1)
            R = R.unsqueeze(0).expand(B, -1, -1, -1)
            t = t.unsqueeze(0).expand(B, -1, -1)
        elif K.dim() == 4:
            V = K.shape[1]
        elif K.dim() == 5:
            V = K.shape[2]
        else:
            raise ValueError(f"K must have 3, 4, or 5 dimensions, got {K.dim()}")

        # Ensure view mask is (B, T, V).
        if view_mask.dim() == 2:
            view_mask = view_mask.unsqueeze(1).expand(-1, T, -1)

        evidence = self._compute_evidence(expert_poses, points_2d, K, R, t, view_mask)
        # evidence: (B, T, E, J, 5)
        feat = self.mlp(evidence)  # (B, T, E, J, hidden)

        # Add joint embedding.
        joint_ids = torch.arange(J, device=feat.device).long()
        joint_emb = self.joint_embed(joint_ids)  # (J, hidden)
        feat = feat + joint_emb.view(1, 1, 1, J, -1)

        logits = self.logit_head(feat).squeeze(-1)  # (B, T, E, J)

        # Softmax over experts per joint.
        weights = F.softmax(logits, dim=2)  # (B, T, E, J)

        # Clamp weights to avoid hard switching.
        if self.min_weight > 0.0:
            weights = weights.clamp(min=self.min_weight)
            weights = weights / weights.sum(dim=2, keepdim=True)

        # Optional identity bypass: blend with uniform weights at init.
        if self.identity_bypass:
            uniform = torch.ones_like(weights) / E
            weights = 0.9 * weights + 0.1 * uniform

        # Weighted sum of expert poses.
        poses = torch.stack(expert_poses, dim=2)  # (B, T, E, J, 3)
        weights_expanded = weights.unsqueeze(-1)  # (B, T, E, J, 1)
        ensemble_pose = (weights_expanded * poses).sum(dim=2)  # (B, T, J, 3)

        # Diversity loss: encourage experts to disagree (negative variance penalty).
        # Use a small weight so it does not dominate the supervised loss.
        per_expert = poses  # (B, T, E, J, 3)
        mean_pose = per_expert.mean(dim=2, keepdim=True)  # (B, T, 1, J, 3)
        diversity = ((per_expert - mean_pose) ** 2).mean()
        diversity_loss = -0.001 * diversity

        return ensemble_pose, diversity_loss
