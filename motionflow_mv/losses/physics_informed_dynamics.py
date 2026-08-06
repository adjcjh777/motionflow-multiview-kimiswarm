"""Physics-informed skeleton dynamics prior losses.

Provides a composite loss that encourages 3-D skeleton trajectories to be
physically plausible without requiring any extra annotations beyond a
skeleton topology (parent indices) and optional foot-joint ids.

The loss is intentionally weakly-supervised: it operates on the predicted
sequence itself, so it can be dropped into any multi-view 3-D pose training
loop that already produces ``(B, T, J, 3)`` world-coordinate poses.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedSkeletonDynamicsLoss(nn.Module):
    """Composite physics prior over a 3-D pose sequence.

    Parameters
    ----------
    parents:
        Parent-index list of length ``J``; ``-1`` for root joints.
    foot_indices:
        Optional list of foot/toe joint indices used by the ground-contact
        term.  If ``None`` the ground-contact term is disabled.
    weights:
        Per-term scalar weights.  Missing terms default to ``0.0``.
        Supported keys: ``bone``, ``jerk``, ``contact``, ``com``.
    eps:
        Small constant for numerical stability.

    Example
    -------
    >>> loss_fn = PhysicsInformedSkeletonDynamicsLoss(
    ...     parents=[-1, 0, 1, ...],
    ...     foot_indices=[3, 6, 10, 13],
    ...     weights={"bone": 1.0, "jerk": 0.1, "contact": 0.5, "com": 0.2},
    ... )
    >>> loss = loss_fn(pred_3d)  # pred_3d: (B, T, J, 3)
    """

    def __init__(
        self,
        parents: Sequence[int],
        foot_indices: Optional[Sequence[int]] = None,
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.parents = list(parents)
        self.foot_indices = list(foot_indices) if foot_indices is not None else []
        self.eps = eps

        default_weights = {"bone": 1.0, "jerk": 0.1, "contact": 0.5, "com": 0.2}
        if weights is not None:
            default_weights.update(weights)
        self.weights = default_weights

    def forward(self, pred_3d: torch.Tensor) -> torch.Tensor:
        """Compute the composite physics loss.

        Args
        ----
        pred_3d:
            Predicted world-coordinate 3-D joints, shape ``(B, T, J, 3)``.

        Returns
        -------
        Scalar loss tensor.
        """
        if pred_3d.dim() != 4:
            raise ValueError(f"pred_3d must be (B, T, J, 3), got {pred_3d.shape}")

        total_loss = torch.tensor(0.0, device=pred_3d.device, dtype=pred_3d.dtype)
        if self.weights.get("bone", 0.0) > 0.0:
            total_loss = total_loss + self.weights["bone"] * bone_length_temporal_variance(
                pred_3d, self.parents
            )
        if self.weights.get("jerk", 0.0) > 0.0:
            total_loss = total_loss + self.weights["jerk"] * jerk_smoothness_loss(pred_3d)
        if self.weights.get("contact", 0.0) > 0.0 and self.foot_indices:
            total_loss = total_loss + self.weights["contact"] * ground_contact_loss(
                pred_3d, self.foot_indices
            )
        if self.weights.get("com", 0.0) > 0.0:
            total_loss = total_loss + self.weights["com"] * center_of_mass_stability_loss(
                pred_3d
            )

        return total_loss


def bone_length_temporal_variance(
    pred: torch.Tensor,
    parents: Sequence[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize temporal variance of bone lengths (dataset-agnostic)."""
    if pred.shape[-2] != len(parents):
        raise ValueError("parents length must equal number of joints")

    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        bone_vec = pred[..., child, :] - pred[..., parent, :]
        bones.append(bone_vec.norm(dim=-1))

    if len(bones) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # (B, T, Bn)
    lengths = torch.stack(bones, dim=-1)
    # Variance over time; mean over batch and bones.
    return lengths.var(dim=-2).mean()


def jerk_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize third-order finite differences of joint trajectories.

    Minimizing the jerk (third derivative) yields smooth, physically plausible
    motion while still allowing realistic acceleration transients.
    """
    if pred.shape[-3] < 4:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Third-order finite difference along the temporal axis.
    jerk = (
        pred[:, 3:]  # (B, T-3, J, 3)
        - 3.0 * pred[:, 2:-1]
        + 3.0 * pred[:, 1:-2]
        - pred[:, :-3]
    )
    return F.mse_loss(jerk, torch.zeros_like(jerk))


def ground_contact_loss(
    pred: torch.Tensor,
    foot_indices: Sequence[int],
    ground_height: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft ground-contact prior for feet.

    Penalizes vertical velocity of foot joints when the foot is close to the
    ground plane.  If ``ground_height`` is not provided, the lowest joint in
    the clip is used as a proxy for the ground plane.
    """
    if pred.shape[-3] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    foot_positions = pred[:, :, foot_indices, :]  # (B, T, F, 3)

    if ground_height is None:
        # Approximate ground as the lowest vertical coordinate in the clip.
        ground_height = foot_positions[..., 2].min()

    # Vertical distance of each foot from the ground.
    foot_height = foot_positions[:, :, :, 2] - ground_height  # (B, T, F)

    # Vertical velocities along the temporal axis.
    v_vert = foot_positions[:, 1:, :, 2] - foot_positions[:, :-1, :, 2]
    v_vert = v_vert.abs()  # (B, T-1, F)

    # Weight by inverse height: feet closer to ground are penalized more.
    height_mid = (foot_height[:, :-1] + foot_height[:, 1:]) / 2.0
    weight = torch.exp(-height_mid.abs() / 0.05)  # soft window ~5 cm

    return (v_vert * weight).mean()


def center_of_mass_stability_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize high-frequency center-of-mass (COM) jerk.

    The COM of a human body should not undergo large, sudden accelerations;
    this term regularizes the temporal derivatives of the mean joint position.
    """
    if pred.shape[-3] < 4:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # COM trajectory: (B, T, 3)
    com = pred.mean(dim=-2)

    # Third derivative (jerk) of COM.
    com_jerk = com[:, 3:] - 3.0 * com[:, 2:-1] + 3.0 * com[:, 1:-2] - com[:, :-3]

    # Also penalize large vertical accelerations.
    acc = com[:, 2:] - 2.0 * com[:, 1:-1] + com[:, :-2]
    vert_acc = acc[..., 2].abs()

    return com_jerk.pow(2).mean() + 0.1 * vert_acc.mean()


__all__ = [
    "PhysicsInformedSkeletonDynamicsLoss",
    "bone_length_temporal_variance",
    "jerk_smoothness_loss",
    "ground_contact_loss",
    "center_of_mass_stability_loss",
]
