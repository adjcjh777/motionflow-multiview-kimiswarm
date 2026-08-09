"""v53 Physical-Space Calibration (PSC).

Calibrates the v52 uncertainty-weighted triangulation output against physical
invariants (floor plane and canonical bone lengths) using a gated residual
refiner. The module is identity at initialization: the final residual projection
is zero-initialized and the residual gate is initialised to a very small value
(``residual_gate_init=-6.0`` => ``sigmoid(gate) ≈ 0.0025``), so the output
pose equals the input pose until training starts.
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
    """Return a sensible parent array for ``j`` joints.

    Falls back to the well-known H36M/MPI skeletons when ``j`` matches one of
    them, otherwise returns a simple kinematic chain.
    """
    if j == 17:
        return list(H36M_17_PARENTS)
    if j == 28:
        return list(MPI_INF_3DHP_28_PARENTS)
    # Generic simple chain.
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
    child_pos = X[:, :, children, :]  # (B, T, n_bones, 3)
    parent_pos = X[:, :, par, :]  # (B, T, n_bones, 3)
    bone_vecs = child_pos - parent_pos
    lengths = bone_vecs.norm(dim=-1, keepdim=False)  # (B, T, n_bones)
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
    X = pred_3d.unsqueeze(2).expand(-1, -1, V, -1, -1)  # (B, T, V, J, 3)
    X = X.permute(0, 1, 2, 4, 3)  # (B, T, V, 3, J)
    X_cam = torch.matmul(R, X) + t[..., None]  # (B, T, V, 3, J)
    X_cam = X_cam.permute(0, 1, 2, 4, 3)  # (B, T, V, J, 3)
    Z = X_cam[..., 2:3].clamp(min=1e-6)
    X_norm = X_cam / Z
    uv = torch.matmul(K[..., None, :, :], X_norm[..., None]).squeeze(-1)  # (B, T, V, J, 3)
    uv = uv[..., :2] / uv[..., 2:3]
    residual = (uv - points_2d).norm(dim=-1)  # (B, T, V, J)
    if view_mask is not None:
        residual = residual * view_mask.unsqueeze(-1).float()
    return residual


class PhysicalSpaceCalibrationV53(nn.Module):
    """Physical-space calibration head for v52 UWT output.

    Parameters
    ----------
    j:
        Number of joints.
    n_views:
        Number of camera views (used only for shape hints).
    hidden:
        Hidden dimension of the residual MLP.
    n_layers:
        Number of layers in the residual MLP.
    num_domains:
        Maximum number of domains for per-domain canonical bone lengths.
    use_floor:
        Enable the floor-plane calibration head.
    use_bone_scale:
        Enable the canonical bone-length calibration head.
    use_uwt_weights:
        Use the v52 UWT weights as a robustness signal for floor estimation.
    identity_init:
        Zero-initialize the final residual layer.
    residual_gate_init:
        Initial value of the residual gate logit.
    floor_joint_indices:
        Optional list of foot/ankle joint indices. If None, the floor head uses
        the lowest joints by gravity coordinate.
    parent_indices:
        Parent array for the bone-length head. If None, inferred from ``j``.
    floor_weight:
        Loss weight for the floor term.
    bone_weight:
        Loss weight for the bone-length term.
    reproj_weight:
        Loss weight for the reprojection consistency term.
    min_visible_views:
        Minimum number of visible views for a joint to contribute to PSC losses.
    """

    def __init__(
        self,
        j: int = 17,
        n_views: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        num_domains: int = 8,
        use_floor: bool = True,
        use_bone_scale: bool = True,
        use_uwt_weights: bool = True,
        identity_init: bool = True,
        residual_gate_init: float = -6.0,
        floor_joint_indices: Optional[List[int]] = None,
        parent_indices: Optional[Sequence[int]] = None,
        floor_weight: float = 0.01,
        bone_weight: float = 0.1,
        reproj_weight: float = 0.1,
        min_visible_views: int = 2,
    ) -> None:
        super().__init__()
        self.j = j
        self.n_views = n_views
        self.use_floor = use_floor
        self.use_bone_scale = use_bone_scale
        self.use_uwt_weights = use_uwt_weights
        self.identity_init = identity_init
        self.floor_weight = floor_weight
        self.bone_weight = bone_weight
        self.reproj_weight = reproj_weight
        self.min_visible_views = max(1, min_visible_views)

        if parent_indices is None:
            parent_indices = _default_parents_for_joints(j)
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long))

        n_bones = sum(1 for p in parent_indices if p >= 0)

        # Floor head: estimate a per-batch/scene floor height.
        # It takes the lowest joints and predicts a translation that puts the
        # feet on the floor. Use a small MLP so the floor height can be smooth.
        self.floor_joint_indices = floor_joint_indices
        if self.use_floor:
            self.floor_mlp = nn.Sequential(
                nn.Linear(2 if use_uwt_weights else 1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            # Zero-init the floor translation so it starts at 0.
            nn.init.zeros_(self.floor_mlp[-1].weight)
            nn.init.zeros_(self.floor_mlp[-1].bias)
        else:
            self.floor_mlp = None  # type: ignore[assignment]

        # Bone-length head: learned per-domain canonical bone lengths.
        # Stored as log-lengths so the initial canonical length is 1.0 in the
        # normalized bone-length space; the predicted scale starts at 1.0.
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

        # Gated physical residual MLP.
        in_dim = 3 + 2  # per-joint 3D position + floor/bone feature hints (2)
        layers: List[nn.Module] = []
        for i in range(n_layers):
            is_last = i == n_layers - 1
            out_dim = 3 if is_last else hidden
            layers.append(nn.Linear(in_dim if i == 0 else hidden, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
        self.residual_mlp = nn.Sequential(*layers)
        if identity_init:
            # Final layer zero-initialized => residual starts at 0.
            final_layer = self.residual_mlp[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

        self.residual_gate = nn.Parameter(torch.tensor(residual_gate_init, dtype=torch.float))

        # Gravity direction. Y-up is the convention in this codebase.
        self.register_buffer("gravity_dir", torch.tensor([0.0, 1.0, 0.0]))

    def _select_floor_joints(self, X: torch.Tensor) -> torch.Tensor:
        """Return foot-joint indices. Either fixed or auto-lowest by gravity."""
        if self.floor_joint_indices is not None:
            return torch.tensor(self.floor_joint_indices, device=X.device, dtype=torch.long)
        # Auto-select the lowest joints along the gravity direction by averaging
        # over the batch/time and choosing the four lowest joints.
        g = self.gravity_dir.to(X.device, X.dtype)  # (3,)
        heights = torch.einsum("btjc,c->btj", X, g)  # (B, T, J)
        mean_heights = heights.mean(dim=(0, 1))  # (J,)
        k = max(2, min(4, self.j))
        floor_ids = torch.topk(mean_heights, k, largest=False).indices  # (k,)
        return floor_ids

    def _floor_term(
        self,
        X: torch.Tensor,
        uwt_weights: Optional[torch.Tensor],
        view_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate floor height and return floor loss + floor translation.

        Args:
            X: (B, T, J, 3) current pose.
            uwt_weights: optional (B, T, V, J) robustness weights.
            view_mask: optional (B, T, V) bool mask.

        Returns:
            floor_loss: scalar.
            floor_translation: (B, T, 1, 3) translation to apply.
        """
        B, T, J, _ = X.shape
        floor_ids = self._select_floor_joints(X)
        if floor_ids.numel() > J:
            floor_ids = floor_ids[:J]

        # Heights of candidate foot joints along gravity.
        g = self.gravity_dir.to(X.device, X.dtype)
        heights = torch.einsum("btjc,c->btj", X, g)  # (B, T, J)
        foot_heights = heights[:, :, floor_ids]  # (B, T, n_feet)

        if self.use_uwt_weights and uwt_weights is not None:
            # Mask views before summing to get per-joint robustness weights.
            w = uwt_weights
            if view_mask is not None:
                w = w * view_mask.unsqueeze(-1).float()
            w = w.sum(dim=2)  # (B, T, J)
            foot_weights = w[:, :, floor_ids]
            # Robust floor estimate: take the lowest foot joint, but weight by
            # visibility/uncertainty.
            feat = torch.stack([foot_heights.min(dim=-1).values, foot_weights.mean(dim=-1)], dim=-1)
        else:
            feat = foot_heights.min(dim=-1).values.unsqueeze(-1)  # (B, T, 1)

        # Predict a per-batch floor offset.
        floor_offset = self.floor_mlp(feat).squeeze(-1)  # (B, T)
        # Floor height is the lowest foot height plus a learned small offset.
        floor_height = foot_heights.min(dim=-1).values + floor_offset
        floor_height = floor_height.clamp(min=-1.0, max=1.0)

        # Soft floor loss: penalise feet below the floor plane.
        floor_plane = floor_height.unsqueeze(-1)  # (B, T, 1)
        violation = (floor_plane - foot_heights).clamp(min=0.0)  # (B, T, n_feet)
        loss = violation.mean()

        # Build a per-batch translation that shifts the floor to y=0.
        # We apply a soft, bounded translation rather than a hard floor clamp
        # to keep the module differentiable and identity-like at init.
        translation = -floor_height.unsqueeze(-1).unsqueeze(-1) * g.view(1, 1, 1, 3)
        # Scale the translation by a small factor so it is a gentle calibration.
        translation = translation * 0.01
        return loss, translation

    def _bone_scale_term(
        self,
        X: torch.Tensor,
        domain_id: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonical bone-length term.

        Args:
            X: (B, T, J, 3)
            domain_id: optional (B,) integer domain labels.

        Returns:
            bone_loss: scalar.
            bone_scale: (B, T, n_bones) predicted per-bone scale.
        """
        B, T, _, _ = X.shape
        _, current_lengths = _compute_bone_vectors_and_lengths(X, self.parent_indices.tolist())
        # current_lengths: (B, T, n_bones)
        n_bones = current_lengths.shape[-1]

        if domain_id is not None and domain_id.numel() > 0:
            d = domain_id.long().clamp(0, self.canonical_log_bone_lengths.shape[0] - 1)
            canonical_lengths = self.canonical_log_bone_lengths[d]  # (B, n_bones)
        else:
            canonical_lengths = self.canonical_log_bone_lengths[0].unsqueeze(0).expand(B, -1)

        # Clamp log-lengths to avoid unbounded exponential growth.
        canonical_lengths = canonical_lengths.clamp(min=-3.0, max=3.0)

        # Canonical target: exp(log) -> start at 1.0; we interpret it as a
        # multiplicative scale target after normalising by the current mean.
        target_scale = torch.exp(canonical_lengths)  # (B, n_bones)
        target_scale = target_scale.unsqueeze(1).expand(-1, T, -1)  # (B, T, n_bones)

        # Predict a soft residual scale around 1.0.
        mean_length = current_lengths.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = current_lengths / mean_length
        feat = torch.cat([normalized, target_scale], dim=-1)  # (B, T, n_bones*2)
        scale_residual = self.bone_scale_mlp(feat)  # (B, T, n_bones)
        # Soft residual: scale in (0.5, 1.5) at init, grows as training demands.
        bone_scale = 1.0 + 0.5 * scale_residual
        bone_scale = bone_scale.clamp(min=0.5, max=2.0)

        # Loss: encourage current length to approach canonical scaled length.
        scaled_target = target_scale * mean_length
        bone_loss = F.mse_loss(current_lengths, scaled_target)
        return bone_loss, bone_scale

    def forward(
        self,
        pred_3d_uwt: torch.Tensor,
        uwt_weights: torch.Tensor,
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calibrate the input 3-D pose against physical invariants.

        Args:
            pred_3d_uwt: (B, T, J, 3) pose from v52 UWT.
            uwt_weights: (B, T, V, J) UWT weights.
            points_2d: (B, T, V, J, 2)
            K: (B, T, V, 3, 3)
            R: (B, T, V, 3, 3)
            t: (B, T, V, 3)
            view_mask: optional (B, T, V) bool mask.
            domain_id: optional (B,) integer domain labels.

        Returns:
            pred_3d_psc: (B, T, J, 3) calibrated pose.
            psc_loss: scalar auxiliary loss.
            floor_height: (B, T) estimated floor height.
            bone_scale: (B, T, n_bones) per-bone scale ratios.
        """
        B, T, J, _ = pred_3d_uwt.shape
        device = pred_3d_uwt.device
        dtype = pred_3d_uwt.dtype

        floor_loss = torch.tensor(0.0, device=device, dtype=dtype)
        bone_loss = torch.tensor(0.0, device=device, dtype=dtype)
        floor_translation = torch.zeros(B, T, 1, 3, device=device, dtype=dtype)
        floor_height = torch.zeros(B, T, device=device, dtype=dtype)
        bone_scale = torch.ones(B, T, self.canonical_log_bone_lengths.shape[-1], device=device, dtype=dtype)

        if self.use_floor:
            floor_loss, floor_translation = self._floor_term(pred_3d_uwt, uwt_weights, view_mask)

        if self.use_bone_scale:
            bone_loss, bone_scale = self._bone_scale_term(pred_3d_uwt, domain_id)

        # Build per-joint feature for residual MLP.
        # Include a floor-distance hint and a bone-scale hint.
        g = self.gravity_dir.to(device, dtype)
        heights = torch.einsum("btjc,c->btj", pred_3d_uwt, g)  # (B, T, J)
        # Broadcast floor_height to per-joint.
        floor_hint = heights - floor_height.unsqueeze(-1)  # (B, T, J)
        bone_hint = bone_scale.mean(dim=-1, keepdim=True).expand(-1, -1, J)  # (B, T, J)
        feat = torch.cat([pred_3d_uwt, floor_hint.unsqueeze(-1), bone_hint.unsqueeze(-1)], dim=-1)

        # Gated residual.
        residual = self.residual_mlp(feat)  # (B, T, J, 3)
        gate = torch.sigmoid(self.residual_gate)
        pred_3d_psc = pred_3d_uwt + gate * residual

        # Reprojection consistency: calibrated pose should not move far from
        # the input where views are visible.
        reproj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.reproj_weight > 0.0:
            with torch.no_grad():
                residual_in = _reprojection_residual(pred_3d_uwt, points_2d, K, R, t, view_mask)
                target = torch.exp(-residual_in / 5.0).clamp(min=0.05, max=1.0)
            pred_residual = _reprojection_residual(pred_3d_psc, points_2d, K, R, t, view_mask)
            weights = target if view_mask is None else target * view_mask.unsqueeze(-1).float()
            reproj_loss = F.mse_loss(pred_residual * weights, residual_in.detach() * weights)

        psc_loss = (
            self.floor_weight * floor_loss
            + self.bone_weight * bone_loss
            + self.reproj_weight * reproj_loss
        )

        return pred_3d_psc, psc_loss, floor_height.detach(), bone_scale.detach()
