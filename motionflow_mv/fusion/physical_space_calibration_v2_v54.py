"""v54 Physical-Space Calibration v2 (PSC-v2).

A skeleton-graph, joint-level physical refiner that sits on top of v53 PSC.
It consumes the v53-calibrated pose and v52 UWT weights, and enforces
floor/contact, per-domain canonical bone-length, and temporal-continuity
constraints while remaining identity at initialization.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.prototypes.cross_view_graph_attention import (
    H36M_17_PARENTS,
    MPI_INF_3DHP_28_PARENTS,
)


def _default_parents_for_joints(j: int) -> List[int]:
    """Return a sensible parent array for ``j`` joints."""
    if j == 17:
        return list(H36M_17_PARENTS)
    if j == 28:
        return list(MPI_INF_3DHP_28_PARENTS)
    return [-1] + list(range(j - 1))


def _compute_bone_vectors_and_lengths(
    X: torch.Tensor, parents: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute bone vectors and lengths from a pose.

    Args:
        X: (B, T, J, 3)
        parents: (J,) parent index; -1 means no parent.

    Returns:
        bone_vecs: (B, T, n_bones, 3)
        lengths:   (B, T, n_bones)
    """
    children: List[int] = []
    par: List[int] = []
    for c, p in enumerate(parents):
        if p >= 0:
            children.append(c)
            par.append(p)
    if not children:
        B, T, _, _ = X.shape
        empty = X.new_zeros(B, T, 0, 3)
        return empty, X.new_zeros(B, T, 0)
    child_pos = X[:, :, children, :]
    parent_pos = X[:, :, par, :]
    bone_vecs = child_pos - parent_pos
    lengths = bone_vecs.norm(dim=-1, keepdim=False)
    return bone_vecs, lengths


def _reprojection_residual(
    pred_3d: torch.Tensor,
    points_2d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    view_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-view per-joint reprojection residual norm (B, T, V, J)."""
    B, T, V, J, _ = points_2d.shape
    X = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)
    X = X.permute(0, 1, 2, 4, 3)
    X_cam = torch.matmul(R, X) + t[..., None]
    X_cam = X_cam.permute(0, 1, 2, 4, 3)
    Z = X_cam[..., 2:3].clamp(min=1e-6)
    X_norm = X_cam / Z
    uv = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)
    uv = uv[..., :2] / uv[..., 2:3]
    residual = (uv - points_2d).norm(dim=-1)
    if view_mask is not None:
        residual = residual * view_mask.unsqueeze(-1).float()
    return residual


def _joint_velocity(X: torch.Tensor) -> torch.Tensor:
    """Return per-joint velocity (B, T-1, J, 3) from pose (B, T, J, 3)."""
    return X[:, 1:] - X[:, :-1]


class PhysicalSpaceCalibrationV2V54(nn.Module):
    """Skeleton-graph physical refiner on top of v53 PSC.

    The module is identity at initialization: the final GNN/MLP residual
    projection, the bone-scale output, and the residual gate are all
    zero-initialized, so ``pred_3d_psc2 == pred_3d_psc`` until training
    starts.
    """

    def __init__(
        self,
        j: int = 17,
        n_views: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        num_domains: int = 8,
        use_floor: bool = True,
        use_contact: bool = True,
        use_bone_scale: bool = True,
        use_temporal_smoothness: bool = True,
        use_gnn: bool = True,
        gnn_layers: int = 1,
        identity_init: bool = True,
        residual_gate_init: float = -6.0,
        floor_joint_indices: Optional[List[int]] = None,
        parent_indices: Optional[Sequence[int]] = None,
        floor_weight: float = 0.01,
        bone_weight: float = 0.05,
        contact_weight: float = 0.01,
        temporal_weight: float = 0.01,
        reproj_weight: float = 0.1,
        contact_velocity_thresh: float = 0.3,
        min_visible_views: int = 2,
    ) -> None:
        super().__init__()
        self.j = j
        self.n_views = n_views
        self.hidden = hidden
        self.use_floor = use_floor
        self.use_contact = use_contact
        self.use_bone_scale = use_bone_scale
        self.use_temporal_smoothness = use_temporal_smoothness
        self.use_gnn = use_gnn
        self.gnn_layers = gnn_layers
        self.identity_init = identity_init
        self.floor_weight = floor_weight
        self.bone_weight = bone_weight
        self.contact_weight = contact_weight
        self.temporal_weight = temporal_weight
        self.reproj_weight = reproj_weight
        self.contact_velocity_thresh = contact_velocity_thresh
        self.min_visible_views = max(1, min_visible_views)

        if parent_indices is None:
            parent_indices = _default_parents_for_joints(j)
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long))

        n_bones = sum(1 for p in parent_indices if p >= 0)

        # Floor head: predicts a per-frame floor height from UWT-weighted feet.
        self.floor_joint_indices = floor_joint_indices
        if self.use_floor:
            self.floor_mlp = nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            nn.init.zeros_(self.floor_mlp[-1].weight)
            nn.init.zeros_(self.floor_mlp[-1].bias)
        else:
            self.floor_mlp = None  # type: ignore[assignment]

        # Per-domain canonical bone log-scales.
        if self.use_bone_scale:
            self.canonical_log_bone_lengths = nn.Parameter(torch.zeros(num_domains, n_bones))
            self.bone_scale_mlp = nn.Sequential(
                nn.Linear(n_bones * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_bones),
                nn.Tanh(),
            )
        else:
            self.register_buffer("canonical_log_bone_lengths", torch.zeros(1, n_bones))
            self.bone_scale_mlp = None  # type: ignore[assignment]

        # Skeleton-graph refiner.
        if self.use_gnn:
            self.gnn_mlps = nn.ModuleList()
            for _ in range(gnn_layers):
                # Node features: position (3) + floor hint (1) + bone-scale hint (1)
                mlp = nn.Sequential(
                    nn.Linear(5 + 5, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 5),
                )
                self.gnn_mlps.append(mlp)
            self.gnn_residual_proj = nn.Linear(5, 3)
        else:
            self.gnn_mlps = nn.ModuleList()
            in_dim = 3 + 1 + 1
            layers: List[nn.Module] = []
            for i in range(n_layers):
                is_last = i == n_layers - 1
                out_dim = 3 if is_last else hidden
                layers.append(nn.Linear(in_dim if i == 0 else hidden, out_dim))
                if not is_last:
                    layers.append(nn.ReLU())
            self.mlp_refiner = nn.Sequential(*layers)
            self.gnn_residual_proj = nn.Linear(3, 3)

        if self.identity_init:
            if self.use_gnn:
                nn.init.zeros_(self.gnn_residual_proj.weight)
                nn.init.zeros_(self.gnn_residual_proj.bias)
            else:
                final_layer = self.mlp_refiner[-1]
                assert isinstance(final_layer, nn.Linear)
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
                nn.init.zeros_(self.gnn_residual_proj.weight)
                nn.init.zeros_(self.gnn_residual_proj.bias)

        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float))
        self.register_buffer("gravity_dir", torch.tensor([0.0, 1.0, 0.0]))

    def _select_floor_joints(self, X: torch.Tensor) -> torch.Tensor:
        """Return foot-joint indices."""
        if self.floor_joint_indices is not None:
            return torch.tensor(self.floor_joint_indices, device=X.device, dtype=torch.long)
        g = self.gravity_dir.to(X.device, X.dtype)
        heights = torch.einsum("btjc,c->btj", X, g)
        mean_heights = heights.mean(dim=(0, 1))
        k = max(2, min(4, self.j))
        return torch.topk(mean_heights, k, largest=False).indices

    def _floor_and_contact_term(
        self,
        X: torch.Tensor,
        uwt_weights: Optional[torch.Tensor],
        view_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate floor height and compute floor + contact losses.

        Returns:
            floor_loss: scalar.
            contact_loss: scalar.
            floor_height: (B, T)
        """
        B, T, J, _ = X.shape
        floor_ids = self._select_floor_joints(X)
        if floor_ids.numel() > J:
            floor_ids = floor_ids[:J]

        g = self.gravity_dir.to(X.device, X.dtype)
        heights = torch.einsum("btjc,c->btj", X, g)
        foot_heights = heights[:, :, floor_ids]

        if uwt_weights is not None:
            w = uwt_weights
            if view_mask is not None:
                w = w * view_mask.unsqueeze(-1).float()
            w = w.sum(dim=2)
            foot_weights = w[:, :, floor_ids]
            feat = torch.stack([foot_heights.min(dim=-1).values, foot_weights.mean(dim=-1)], dim=-1)
        else:
            feat = foot_heights.min(dim=-1).values.unsqueeze(-1)

        floor_offset = self.floor_mlp(feat).squeeze(-1)
        floor_height = foot_heights.min(dim=-1).values + floor_offset
        floor_height = floor_height.clamp(min=-1.0, max=1.0)

        # Soft floor loss: penalise feet below the floor plane.
        violation = (floor_height.unsqueeze(-1) - foot_heights).clamp(min=0.0)
        floor_loss = violation.mean()

        contact_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if self.use_contact and T > 1:
            # Foot joints in world coordinates (B, T, n_feet, 3)
            foot_pos = X[:, :, floor_ids, :]
            # Height above floor.
            foot_above_floor = foot_pos @ g  # (B, T, n_feet)
            # Velocity of each foot joint.
            if T > 1:
                vel = _joint_velocity(foot_pos)  # (B, T-1, n_feet, 3)
                speed = vel.norm(dim=-1)  # (B, T-1, n_feet)
                # Use mid-frame floor height.
                fh_mid = 0.5 * (floor_height[:, 1:] + floor_height[:, :-1])
                above_mid = foot_above_floor[:, 1:, :]
                # Contact mask: slow feet should stay close to the floor.
                contact_mask = (speed < self.contact_velocity_thresh).float()
                contact_residual = (above_mid - fh_mid.unsqueeze(-1)).clamp(min=0.0)
                contact_loss = (contact_mask * contact_residual).mean()

        return floor_loss, contact_loss, floor_height

    def _bone_scale_term(
        self,
        X: torch.Tensor,
        domain_id: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical bone-length term.

        Returns:
            bone_loss: scalar.
            bone_scale: (B, T, n_bones)
        """
        B, T, _, _ = X.shape
        _, current_lengths = _compute_bone_vectors_and_lengths(X, self.parent_indices.tolist())
        n_bones = current_lengths.shape[-1]

        if domain_id is not None and domain_id.numel() > 0:
            d = domain_id.long().clamp(0, self.canonical_log_bone_lengths.shape[0] - 1)
            canonical_lengths = self.canonical_log_bone_lengths[d]
        else:
            canonical_lengths = self.canonical_log_bone_lengths[0].unsqueeze(0).expand(B, -1)

        target_scale = torch.exp(canonical_lengths)
        target_scale = target_scale.unsqueeze(1).expand(-1, T, -1)

        mean_length = current_lengths.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = current_lengths / mean_length
        feat = torch.cat([normalized, target_scale], dim=-1)
        scale_residual = self.bone_scale_mlp(feat)
        bone_scale = 1.0 + 0.5 * scale_residual
        bone_scale = bone_scale.clamp(min=0.5, max=2.0)

        scaled_target = target_scale * mean_length
        bone_loss = F.mse_loss(current_lengths, scaled_target)
        return bone_loss, bone_scale

    def _gnn_residual(
        self,
        X_in: torch.Tensor,
        floor_hint: torch.Tensor,
        bone_hint: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a gated, identity-initialised per-joint residual.

        Args:
            X_in: (B, T, J, 3)
            floor_hint: (B, T, J)
            bone_hint: (B, T, J)

        Returns:
            residual: (B, T, J, 3)
        """
        B, T, J, _ = X_in.shape
        device, dtype = X_in.device, X_in.dtype

        if self.use_gnn:
            # Node feature: [position, floor_hint, bone_hint]
            feat = torch.cat([X_in, floor_hint.unsqueeze(-1), bone_hint.unsqueeze(-1)], dim=-1)
            for layer in self.gnn_mlps:
                # Message passing over the kinematic tree.
                messages = torch.zeros_like(feat)
                parents = self.parent_indices.tolist()
                for child, parent in enumerate(parents):
                    if parent >= 0:
                        messages[:, :, child, :] += feat[:, :, parent, :]
                        messages[:, :, parent, :] += feat[:, :, child, :]
                # Update with self + neighbor messages.
                update_input = torch.cat([feat, messages], dim=-1)
                feat = feat + layer(update_input)
            residual = self.gnn_residual_proj(feat)
        else:
            feat = torch.cat([X_in, floor_hint.unsqueeze(-1), bone_hint.unsqueeze(-1)], dim=-1)
            residual = self.mlp_refiner(feat)
            residual = self.gnn_residual_proj(residual)

        gate = torch.sigmoid(self.residual_gate)
        return gate * residual

    def forward(
        self,
        pred_3d_psc: torch.Tensor,
        uwt_weights: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine the v53-calibrated pose.

        Args:
            pred_3d_psc: (B, T, J, 3) pose from v53.
            uwt_weights: (B, T, V, J) v52 UWT weights.
            points_2d: (B, T, V, J, 2)
            K: (B, T, V, 3, 3)
            R: (B, T, V, 3, 3)
            t: (B, T, V, 3)
            view_mask: optional (B, T, V) bool mask.
            domain_id: optional (B,) integer domain labels.

        Returns:
            pred_3d_psc2: (B, T, J, 3)
            psc2_loss: scalar
            floor_height_v2: (B, T)
            bone_scale_v2: (B, T, n_bones)
        """
        B, T, J, _ = pred_3d_psc.shape
        device, dtype = pred_3d_psc.device, pred_3d_psc.dtype

        floor_loss = torch.tensor(0.0, device=device, dtype=dtype)
        contact_loss = torch.tensor(0.0, device=device, dtype=dtype)
        bone_loss = torch.tensor(0.0, device=device, dtype=dtype)
        temporal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        floor_height_v2 = torch.zeros(B, T, device=device, dtype=dtype)
        bone_scale_v2 = torch.ones(B, T, self.canonical_log_bone_lengths.shape[-1], device=device, dtype=dtype)

        if self.use_floor:
            floor_loss, contact_loss, floor_height_v2 = self._floor_and_contact_term(
                pred_3d_psc, uwt_weights, view_mask
            )

        if self.use_bone_scale:
            bone_loss, bone_scale_v2 = self._bone_scale_term(pred_3d_psc, domain_id)

        # Build per-joint hints for the refiner.
        g = self.gravity_dir.to(device, dtype)
        heights = torch.einsum("btjc,c->btj", pred_3d_psc, g)
        floor_hint = heights - floor_height_v2.unsqueeze(-1)
        bone_hint = bone_scale_v2.mean(dim=-1, keepdim=True).expand(-1, -1, J)

        residual = self._gnn_residual(pred_3d_psc, floor_hint, bone_hint)
        pred_3d_psc2 = pred_3d_psc + residual

        if self.use_temporal_smoothness and T > 2:
            correction = pred_3d_psc2 - pred_3d_psc
            # Second-order finite difference over time.
            acc = correction[:, 2:] - 2 * correction[:, 1:-1] + correction[:, :-2]
            temporal_loss = acc.pow(2).mean()

        reproj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.reproj_weight > 0.0:
            with torch.no_grad():
                residual_in = _reprojection_residual(pred_3d_psc, points_2d, K, R, t, view_mask)
                target = torch.exp(-residual_in / 5.0).clamp(min=0.05, max=1.0)
            pred_residual = _reprojection_residual(pred_3d_psc2, points_2d, K, R, t, view_mask)
            weights = target if view_mask is None else target * view_mask.unsqueeze(-1).float()
            reproj_loss = F.mse_loss(pred_residual * weights, residual_in.detach() * weights)

        psc2_loss = (
            self.floor_weight * floor_loss
            + (self.contact_weight * contact_loss if self.use_contact else 0.0)
            + self.bone_weight * bone_loss
            + self.temporal_weight * temporal_loss
            + self.reproj_weight * reproj_loss
        )

        return pred_3d_psc2, psc2_loss, floor_height_v2.detach(), bone_scale_v2.detach()
