"""v57 Domain-Conditional Physical-Space Calibration (DC-PSC).

Extends v53 Physical-Space Calibration with per-domain conditioning so that
floor/bone/residual calibration can adapt to WebBridge, H36M, MPI, 3DPW, etc.
The module remains identity at initialization: the final residual projection is
zero-initialized and the residual gate is initialised to a very small value
(``residual_gate_init=-6.0`` => ``sigmoid(gate) ≈ 0.0025``), so the output pose
equals the input pose until training starts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionflow_mv.fusion.physical_space_calibration_v53 import (
    _compute_bone_vectors_and_lengths,
    _default_parents_for_joints,
    _reprojection_residual,
)


class DomainConditionalPhysicalCalibrationV57(nn.Module):
    """Domain-conditional physical-space calibration head for v52 UWT output.

    Parameters
    ----------
    j:
        Number of joints.
    n_views:
        Number of camera views (used only for shape hints).
    hidden:
        Hidden dimension of the FiLM-conditioned MLPs.
    n_layers:
        Number of layers in the residual MLP.
    num_domains:
        Maximum number of domains for the shared domain embedding, per-domain
        canonical bone lengths, per-domain residual gate, etc.
    use_floor:
        Enable the domain-conditional floor-plane calibration head.
    use_bone_scale:
        Enable the domain-conditional canonical bone-length calibration head.
    use_uwt_weights:
        Use the v52 UWT weights as a robustness signal for floor estimation.
    identity_init:
        Zero-initialize the final layers of floor, bone and residual MLPs.
    residual_gate_init:
        Initial value of the per-domain residual gate logits.
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
        floor_weight: float = 0.01,
        bone_weight: float = 0.1,
        reproj_weight: float = 0.1,
        min_visible_views: int = 2,
        stop_grad_to_base: bool = False,
        max_correction: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.j = j
        self.n_views = n_views
        self.hidden = hidden
        self.n_layers = max(1, n_layers)
        self.num_domains = num_domains
        self.use_floor = use_floor
        self.use_bone_scale = use_bone_scale
        self.use_uwt_weights = use_uwt_weights
        self.identity_init = identity_init
        self.floor_weight = floor_weight
        self.bone_weight = bone_weight
        self.reproj_weight = reproj_weight
        self.min_visible_views = max(1, min_visible_views)
        self.stop_grad_to_base = stop_grad_to_base
        self.max_correction = max_correction

        parent_indices = _default_parents_for_joints(j)
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long))
        n_bones = sum(1 for p in parent_indices if p >= 0)

        # Shared domain embedding used by all three heads.
        self.domain_embedding = nn.Embedding(num_domains, hidden)
        nn.init.normal_(self.domain_embedding.weight, 0.0, 0.01)

        # Floor head: estimate a per-batch/scene floor height conditioned on domain.
        if self.use_floor:
            self.floor_first = nn.Linear((2 if use_uwt_weights else 1) + hidden, hidden)
            self.floor_film_scale = nn.Linear(hidden, hidden)
            self.floor_film_shift = nn.Linear(hidden, hidden)
            self.floor_mid = nn.Linear(hidden, hidden // 2)
            self.floor_out = nn.Linear(hidden // 2, 1)
            nn.init.zeros_(self.floor_film_scale.weight)
            nn.init.zeros_(self.floor_film_scale.bias)
            nn.init.zeros_(self.floor_film_shift.weight)
            nn.init.zeros_(self.floor_film_shift.bias)
            if identity_init:
                nn.init.zeros_(self.floor_out.weight)
                nn.init.zeros_(self.floor_out.bias)
        else:
            self.floor_first = None  # type: ignore[assignment]

        # Bone-length head: learned per-domain canonical bone lengths.
        if self.use_bone_scale:
            self.canonical_log_bone_lengths = nn.Parameter(torch.zeros(num_domains, n_bones))
            self.bone_first = nn.Linear(n_bones * 2 + hidden, hidden)
            self.bone_film_scale = nn.Linear(hidden, hidden)
            self.bone_film_shift = nn.Linear(hidden, hidden)
            self.bone_mid = nn.Linear(hidden, hidden)
            self.bone_out = nn.Linear(hidden, n_bones)
            nn.init.zeros_(self.bone_film_scale.weight)
            nn.init.zeros_(self.bone_film_scale.bias)
            nn.init.zeros_(self.bone_film_shift.weight)
            nn.init.zeros_(self.bone_film_shift.bias)
            if identity_init:
                nn.init.zeros_(self.bone_out.weight)
                nn.init.zeros_(self.bone_out.bias)
        else:
            self.register_buffer("canonical_log_bone_lengths", torch.zeros(1, n_bones))

        # Gated physical residual MLP with domain FiLM conditioning.
        in_dim = 3 + 2 + hidden  # per-joint 3D + floor/bone hints + domain embedding
        self.residual_first = nn.Linear(in_dim, hidden)
        self.residual_hidden_layers = nn.ModuleList()
        self.residual_film_scales = nn.ModuleList()
        self.residual_film_shifts = nn.ModuleList()
        for _ in range(self.n_layers - 1):
            self.residual_hidden_layers.append(nn.Linear(hidden, hidden))
            self.residual_film_scales.append(nn.Linear(hidden, hidden))
            self.residual_film_shifts.append(nn.Linear(hidden, hidden))
        for lin in self.residual_film_scales:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.residual_film_shifts:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.residual_out = nn.Linear(hidden, 3)
        if identity_init:
            nn.init.zeros_(self.residual_out.weight)
            nn.init.zeros_(self.residual_out.bias)

        # Per-domain residual gate logit; small sigmoid at init.
        self.residual_gate = nn.Parameter(
            torch.full((num_domains,), residual_gate_init, dtype=torch.float)
        )

        # Gravity direction (Y-up convention).
        self.register_buffer("gravity_dir", torch.tensor([0.0, 1.0, 0.0]))

    def _select_floor_joints(self, X: torch.Tensor) -> torch.Tensor:
        """Return foot-joint indices. Auto-lowest by gravity direction."""
        g = self.gravity_dir.to(X.device, X.dtype)  # (3,)
        heights = torch.einsum("btjc,c->btj", X, g)  # (B, T, J)
        mean_heights = heights.mean(dim=(0, 1))  # (J,)
        k = max(2, min(4, self.j))
        floor_ids = torch.topk(mean_heights, k, largest=False).indices  # (k,)
        return floor_ids

    def _lookup_domain_embedding(self, domain_id: Optional[torch.Tensor]) -> torch.Tensor:
        """Return (B, hidden) domain embedding; fall back to the mean embedding."""
        if domain_id is not None and domain_id.numel() > 0:
            d = domain_id.long().clamp(0, self.num_domains - 1)
            return self.domain_embedding(d)  # (B, hidden)
        return self.domain_embedding.weight.mean(dim=0, keepdim=True)  # (1, hidden)

    def _film(
        self,
        h: torch.Tensor,
        domain_emb: torch.Tensor,
        scale_layer: nn.Linear,
        shift_layer: nn.Linear,
    ) -> torch.Tensor:
        """Apply FiLM: h = scale * h + shift."""
        scale = scale_layer(domain_emb)
        shift = shift_layer(domain_emb)
        # Broadcast scale/shift over the spatial dimensions of h.
        for _ in range(h.dim() - 2):
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return scale * h + shift

    def _compute_psc_loss(
        self,
        X: torch.Tensor,
        uwt_weights: Optional[torch.Tensor],
        points_2d: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        view_mask: Optional[torch.Tensor],
        domain_id: Optional[torch.Tensor],
        domain_emb: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the v57 PSC auxiliary loss on a detached copy of the input.

        When ``stop_grad_to_base`` is enabled, the main forward path still uses
        the undetached input so earlier modules receive gradients from the final
        MSE loss.  The auxiliary loss, however, is computed on a detached branch,
        so it only updates the v57 head parameters and cannot destabilise the
        v52/v60 stack.
        """
        B, T, J, _ = X.shape
        device, dtype = X.device, X.dtype

        floor_loss = torch.tensor(0.0, device=device, dtype=dtype)
        bone_loss = torch.tensor(0.0, device=device, dtype=dtype)
        bone_scale = torch.ones(
            B, T, self.canonical_log_bone_lengths.shape[-1], device=device, dtype=dtype
        )

        if self.use_floor and self.floor_first is not None:
            floor_loss, _ = self._floor_term(X, uwt_weights, view_mask, domain_emb)

        if self.use_bone_scale:
            bone_loss, bone_scale = self._bone_scale_term(X, domain_id, domain_emb)

        g = self.gravity_dir.to(device, dtype)
        heights = torch.einsum("btjc,c->btj", X, g)
        floor_hint = heights
        bone_hint = bone_scale.mean(dim=-1, keepdim=True).expand(-1, -1, J)
        emb_tiled = domain_emb.unsqueeze(1).unsqueeze(1).expand(-1, T, J, -1)
        feat = torch.cat(
            [X, floor_hint.unsqueeze(-1), bone_hint.unsqueeze(-1), emb_tiled], dim=-1
        )

        h = F.relu(self.residual_first(feat))
        for hidden_layer, scale_layer, shift_layer in zip(
            self.residual_hidden_layers, self.residual_film_scales, self.residual_film_shifts
        ):
            h = self._film(h, domain_emb, scale_layer, shift_layer)
            h = F.relu(hidden_layer(h))
        residual = self.residual_out(h)

        correction = gate.view(-1, 1, 1, 1) * residual
        if self.max_correction is not None:
            correction = correction.clamp(min=-self.max_correction, max=self.max_correction)
        pred_3d_psc = X + correction

        reproj_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.reproj_weight > 0.0:
            with torch.no_grad():
                residual_in = _reprojection_residual(X, points_2d, K, R, t, view_mask)
                target = torch.exp(-residual_in / 5.0).clamp(min=0.05, max=1.0)
            pred_residual = _reprojection_residual(pred_3d_psc, points_2d, K, R, t, view_mask)
            weights = target if view_mask is None else target * view_mask.unsqueeze(-1).float()
            reproj_loss = F.mse_loss(pred_residual * weights, residual_in.detach() * weights)

        return self.floor_weight * floor_loss + self.bone_weight * bone_loss + self.reproj_weight * reproj_loss

    def _floor_term(
        self,
        X: torch.Tensor,
        uwt_weights: Optional[torch.Tensor],
        view_mask: Optional[torch.Tensor],
        domain_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate floor height and return floor loss + floor translation.

        Args:
            X: (B, T, J, 3) current pose.
            uwt_weights: optional (B, T, V, J) robustness weights.
            view_mask: optional (B, T, V) bool mask.
            domain_emb: (B, hidden) domain embedding.

        Returns:
            floor_loss: scalar.
            floor_translation: (B, T, 1, 3) translation to apply.
        """
        B, T, J, _ = X.shape
        floor_ids = self._select_floor_joints(X)
        if floor_ids.numel() > J:
            floor_ids = floor_ids[:J]

        g = self.gravity_dir.to(X.device, X.dtype)
        heights = torch.einsum("btjc,c->btj", X, g)  # (B, T, J)
        foot_heights = heights[:, :, floor_ids]  # (B, T, n_feet)

        if self.use_uwt_weights and uwt_weights is not None:
            w = uwt_weights
            if view_mask is not None:
                w = w * view_mask.unsqueeze(-1).float()
            w = w.sum(dim=2)  # (B, T, J)
            foot_weights = w[:, :, floor_ids]
            feat = torch.stack(
                [foot_heights.min(dim=-1).values, foot_weights.mean(dim=-1)], dim=-1
            )  # (B, T, 2)
        else:
            feat = foot_heights.min(dim=-1).values.unsqueeze(-1)  # (B, T, 1)

        # Concatenate domain embedding to the floor feature.
        emb_tiled = domain_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
        feat = torch.cat([feat, emb_tiled], dim=-1)

        h = F.relu(self.floor_first(feat))
        h = self._film(h, domain_emb, self.floor_film_scale, self.floor_film_shift)
        h = F.relu(self.floor_mid(h))
        floor_offset = self.floor_out(h).squeeze(-1)  # (B, T)

        floor_height = foot_heights.min(dim=-1).values + floor_offset
        floor_height = floor_height.clamp(min=-1.0, max=1.0)

        floor_plane = floor_height.unsqueeze(-1)  # (B, T, 1)
        violation = (floor_plane - foot_heights).clamp(min=0.0)  # (B, T, n_feet)
        loss = violation.mean()

        translation = -floor_height.unsqueeze(-1).unsqueeze(-1) * g.view(1, 1, 1, 3)
        translation = translation * 0.01
        return loss, translation

    def _bone_scale_term(
        self,
        X: torch.Tensor,
        domain_id: Optional[torch.Tensor],
        domain_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Domain-conditional canonical bone-length term.

        Args:
            X: (B, T, J, 3)
            domain_id: optional (B,) integer domain labels.
            domain_emb: (B, hidden) domain embedding.

        Returns:
            bone_loss: scalar.
            bone_scale: (B, T, n_bones) predicted per-bone scale.
        """
        B, T, _, _ = X.shape
        _, current_lengths = _compute_bone_vectors_and_lengths(X, self.parent_indices.tolist())
        n_bones = current_lengths.shape[-1]

        if domain_id is not None and domain_id.numel() > 0:
            d = domain_id.long().clamp(0, self.canonical_log_bone_lengths.shape[0] - 1)
            canonical_lengths = self.canonical_log_bone_lengths[d]  # (B, n_bones)
        else:
            canonical_lengths = self.canonical_log_bone_lengths[0].unsqueeze(0).expand(B, -1)

        # Clamp log-lengths to avoid unbounded exponential growth.
        canonical_lengths = canonical_lengths.clamp(min=-3.0, max=3.0)

        target_scale = torch.exp(canonical_lengths)  # (B, n_bones)
        target_scale = target_scale.unsqueeze(1).expand(-1, T, -1)  # (B, T, n_bones)

        mean_length = current_lengths.mean(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = current_lengths / mean_length
        emb_tiled = domain_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
        feat = torch.cat([normalized, target_scale, emb_tiled], dim=-1)

        h = F.relu(self.bone_first(feat))
        h = self._film(h, domain_emb, self.bone_film_scale, self.bone_film_shift)
        h = F.relu(self.bone_mid(h))
        scale_residual = torch.tanh(self.bone_out(h))  # (B, T, n_bones)

        bone_scale = 1.0 + 0.5 * scale_residual
        bone_scale = bone_scale.clamp(min=0.5, max=2.0)

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

        domain_emb = self._lookup_domain_embedding(domain_id)  # (B, hidden)
        # Defensive: when no domain_id is supplied (e.g. monotonic-loss forward),
        # broadcast the fallback mean embedding to the current batch size.
        if domain_emb.shape[0] != B:
            domain_emb = domain_emb.expand(B, -1)

        floor_loss = torch.tensor(0.0, device=device, dtype=dtype)
        bone_loss = torch.tensor(0.0, device=device, dtype=dtype)
        floor_translation = torch.zeros(B, T, 1, 3, device=device, dtype=dtype)
        floor_height = torch.zeros(B, T, device=device, dtype=dtype)
        bone_scale = torch.ones(
            B, T, self.canonical_log_bone_lengths.shape[-1], device=device, dtype=dtype
        )

        if self.use_floor and self.floor_first is not None:
            floor_loss, floor_translation = self._floor_term(
                pred_3d_uwt, uwt_weights, view_mask, domain_emb
            )

        if self.use_bone_scale:
            bone_loss, bone_scale = self._bone_scale_term(pred_3d_uwt, domain_id, domain_emb)

        # Build per-joint feature for residual MLP.
        g = self.gravity_dir.to(device, dtype)
        heights = torch.einsum("btjc,c->btj", pred_3d_uwt, g)  # (B, T, J)
        floor_hint = heights - floor_height.unsqueeze(-1)  # (B, T, J)
        bone_hint = bone_scale.mean(dim=-1, keepdim=True).expand(-1, -1, J)  # (B, T, J)
        emb_tiled = domain_emb.unsqueeze(1).unsqueeze(1).expand(-1, T, J, -1)  # (B, T, J, hidden)
        feat = torch.cat(
            [pred_3d_uwt, floor_hint.unsqueeze(-1), bone_hint.unsqueeze(-1), emb_tiled], dim=-1
        )  # (B, T, J, in_dim)

        # Gated residual with FiLM conditioning in hidden layers.
        h = F.relu(self.residual_first(feat))
        for hidden_layer, scale_layer, shift_layer in zip(
            self.residual_hidden_layers, self.residual_film_scales, self.residual_film_shifts
        ):
            h = self._film(h, domain_emb, scale_layer, shift_layer)
            h = F.relu(hidden_layer(h))
        residual = self.residual_out(h)  # (B, T, J, 3)

        # Per-domain residual gate.
        if domain_id is not None and domain_id.numel() > 0:
            gate_logit = self.residual_gate[domain_id.long().clamp(0, self.num_domains - 1)]  # (B,)
        else:
            gate_logit = self.residual_gate.mean()
        gate = torch.sigmoid(gate_logit)
        correction = gate.view(-1, 1, 1, 1) * residual
        if self.max_correction is not None:
            correction = correction.clamp(min=-self.max_correction, max=self.max_correction)
        pred_3d_psc = pred_3d_uwt + correction

        # Reprojection consistency.
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

        # If requested, recompute the auxiliary loss on a detached input so that
        # the v57 PSC loss only updates v57 parameters and cannot destabilise the
        # v52 UWT / v60 SEFH->UWT feedback stack.
        if self.stop_grad_to_base and self.training:
            psc_loss = self._compute_psc_loss(
                pred_3d_uwt.detach(),
                uwt_weights,
                points_2d,
                K,
                R,
                t,
                view_mask,
                domain_id,
                domain_emb,
                gate,
            )

        return pred_3d_psc, psc_loss, floor_height.detach(), bone_scale.detach()
