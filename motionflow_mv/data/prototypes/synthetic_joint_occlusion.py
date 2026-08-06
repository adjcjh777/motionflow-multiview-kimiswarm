"""Synthetic joint occlusion augmentation for multi-view 2D keypoints.

The module augments the canonical ``(..., V, J, C)`` tensors by occluding
*groups* of anatomically related joints rather than independent random joints.
This better mimics real occlusions: a limb or the torso tends to disappear as
a whole when it is behind another body part or outside the image border.

API summary
-----------
* :func:`occlude_joint_groups`            -- occlude named joint groups.
* :func:`random_occlude_joint_groups`    -- randomly sample groups to drop.
* :class:`SyntheticJointOcclusionAugmenter` -- configurable drop-in augmenter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Skeleton group definitions
# ---------------------------------------------------------------------------

H36M_17_JOINT_GROUPS: Dict[str, List[int]] = {
    "torso": [0, 7, 8],
    "head": [9, 10],
    "left_arm": [11, 12, 13],
    "right_arm": [14, 15, 16],
    "left_leg": [4, 5, 6],
    "right_leg": [1, 2, 3],
}

MPI_INF_3DHP_28_JOINT_GROUPS: Dict[str, List[int]] = {
    "head": [0, 1, 5, 6, 7],
    "torso": [2, 3, 4],
    "left_arm": [8, 9, 10, 11, 12],
    "right_arm": [13, 14, 15, 16, 17],
    "left_leg": [18, 19, 20, 21, 22],
    "right_leg": [23, 24, 25, 26, 27],
}

# Maps supported skeleton aliases to group table.
_SKELETON_GROUPS = {
    "h36m_17": H36M_17_JOINT_GROUPS,
    "mpiinf3dhp_28": MPI_INF_3DHP_28_JOINT_GROUPS,
}


def _validate_view_joint_dims(x: torch.Tensor) -> Tuple[int, int]:
    if x.dim() < 3:
        raise ValueError(f"Input must have at least 3 dimensions, got shape {x.shape}")
    return int(x.shape[-3]), int(x.shape[-2])


def _resolve_confidence_channel(x: torch.Tensor, confidence_channel: int) -> int:
    c = confidence_channel if confidence_channel >= 0 else x.shape[-1] + confidence_channel
    if c < 0 or c >= x.shape[-1]:
        raise ValueError(
            f"confidence_channel {confidence_channel} out of bounds for last dim {x.shape[-1]}"
        )
    return c


def occlude_joint_groups(
    x: torch.Tensor,
    group_indices: Union[str, Sequence[int], Dict[str, List[int]]],
    group_names: Optional[Sequence[str]] = None,
    confidence_channel: int = -1,
    zero_coords: bool = False,
) -> torch.Tensor:
    """Occlude one or more joint groups across all views.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        group_indices: Either a skeleton alias (``'h36m_17'`` or
            ``'mpiinf3dhp_28'``), a list of joint indices, or a dict mapping a
            group name to a list of joint indices.  If a dict is passed, the
            ``group_names`` argument selects which groups to occlude.
        group_names: Names of groups to occlude.  Required when ``group_indices``
            is a dict; ignored otherwise.
        confidence_channel: Index of the confidence/visibility channel.
        zero_coords: If ``True``, also zero the 2D coordinate channels.

    Returns:
        Occluded tensor with the same shape as ``x``.
    """
    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    if isinstance(group_indices, str):
        group_indices = _SKELETON_GROUPS[group_indices]

    if isinstance(group_indices, dict):
        if group_names is None:
            raise ValueError("group_names is required when group_indices is a dict")
        joint_indices: List[int] = []
        for name in group_names:
            if name in group_indices:
                joint_indices.extend(group_indices[name])
        group_indices = joint_indices
    else:
        group_indices = list(group_indices)

    if not group_indices:
        return x

    joint_indices_t = torch.as_tensor(group_indices, dtype=torch.long, device=x.device)
    joint_indices_t = joint_indices_t[(joint_indices_t >= 0) & (joint_indices_t < J)].unique()
    if joint_indices_t.numel() == 0:
        return x

    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])
    x_view[:, :, joint_indices_t, c] = 0.0
    if zero_coords:
        x_view[:, :, joint_indices_t, :2] = 0.0
    return x


def random_occlude_joint_groups(
    x: torch.Tensor,
    group_rate: float,
    skeleton: Union[str, Dict[str, List[int]]] = "h36m_17",
    per_sample: bool = False,
    confidence_channel: int = -1,
    zero_coords: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly occlude whole joint groups.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        group_rate: Probability of occluding each group (0 <= group_rate <= 1).
        skeleton: Skeleton alias or explicit group dict.
        per_sample: If ``True``, sample groups independently for each leading
            sample (e.g. batch element + frame).  If ``False``, share the same
            group mask across all leading samples.
        confidence_channel: Index of the confidence/visibility channel.
        zero_coords: If ``True``, also zero the 2D coordinate channels.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Occluded tensor.
    """
    if group_rate <= 0.0:
        return x

    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    if isinstance(skeleton, str):
        skeleton = _SKELETON_GROUPS[skeleton]

    group_names = list(skeleton.keys())
    group_list = [skeleton[name] for name in group_names]

    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])

    if per_sample:
        mask = torch.rand(N, len(group_names), generator=generator, device=x.device) < group_rate
    else:
        mask = torch.rand(len(group_names), generator=generator, device=x.device) < group_rate

    for g_idx, joint_indices in enumerate(group_list):
        joint_indices_t = torch.as_tensor(joint_indices, dtype=torch.long, device=x.device)
        joint_indices_t = joint_indices_t[(joint_indices_t >= 0) & (joint_indices_t < J)]
        if joint_indices_t.numel() == 0:
            continue

        if per_sample:
            active = mask[:, g_idx]  # (N,)
            active_c = active.view(-1, 1, 1, 1).expand(-1, V, len(joint_indices_t), 1)
            x_view[:, :, joint_indices_t, c:c + 1] = torch.where(
                active_c,
                0.0,
                x_view[:, :, joint_indices_t, c:c + 1],
            )
            if zero_coords:
                active_xy = active.view(-1, 1, 1, 1).expand(-1, V, len(joint_indices_t), 2)
                x_view[:, :, joint_indices_t, :2] = torch.where(
                    active_xy,
                    0.0,
                    x_view[:, :, joint_indices_t, :2],
                )
        else:
            if mask[g_idx]:
                x_view[:, :, joint_indices_t, c] = 0.0
                if zero_coords:
                    x_view[:, :, joint_indices_t, :2] = 0.0

    return x


class SyntheticJointOcclusionAugmenter:
    """Drop-in synthetic joint occlusion augmenter.

    Combines group-level occlusion with a small amount of independent per-joint
    occlusion so the model also sees isolated joint dropouts.

    Args:
        skeleton: ``'h36m_17'`` or ``'mpiinf3dhp_28'``.
        group_rate: Probability of occluding each body-part group.
        joint_rate: Additional independent per-joint occlusion probability.
        temporal_consistency: If ``True`` and the input has shape
            ``(B, T, V, J, C)``, the same occlusion mask is applied across the
            temporal dimension ``T`` for each sample.  This simulates occlusions
            that persist over short clips.
        zero_coords: If ``True``, also zero the 2D coordinate channels when a
            joint is occluded.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        skeleton: str = "h36m_17",
        group_rate: float = 0.0,
        joint_rate: float = 0.0,
        temporal_consistency: bool = False,
        zero_coords: bool = False,
        seed: Optional[int] = None,
    ):
        if skeleton not in _SKELETON_GROUPS:
            raise ValueError(f"Unsupported skeleton: {skeleton}")
        self.skeleton = skeleton
        self.group_rate = group_rate
        self.joint_rate = joint_rate
        self.temporal_consistency = temporal_consistency
        self.zero_coords = zero_coords
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply synthetic joint occlusion to ``x``.

        The operation is performed on a clone, so the input is left unchanged.
        """
        if self.temporal_consistency and x.dim() == 5:
            x = self._apply_temporal(x)
        else:
            x = random_occlude_joint_groups(
                x,
                self.group_rate,
                skeleton=self.skeleton,
                per_sample=False,
                zero_coords=self.zero_coords,
                generator=self.generator,
            )

        # Independent per-joint dropout for isolated joint failures.
        if self.joint_rate > 0.0:
            from motionflow_mv.data.occlusion_aug import random_occlude_joints

            x = random_occlude_joints(
                x,
                self.joint_rate,
                per_view=True,
                per_sample=False,
                zero_coords=self.zero_coords,
                generator=self.generator,
            )
        return x

    def _apply_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same group occlusion mask across the temporal dimension."""
        B, T, V, J, C = x.shape
        skeleton = _SKELETON_GROUPS[self.skeleton]
        group_names = list(skeleton.keys())
        group_list = [skeleton[name] for name in group_names]
        G = len(group_names)

        # Sample one mask per batch sample (B, G).
        mask = torch.rand(B, G, generator=self.generator, device=x.device) < self.group_rate

        x = x.clone()
        channels = [0, 1] if self.zero_coords else []
        for g_idx, joint_indices in enumerate(group_list):
            joint_indices_t = torch.as_tensor(joint_indices, dtype=torch.long, device=x.device)
            joint_indices_t = joint_indices_t[(joint_indices_t >= 0) & (joint_indices_t < J)]
            if joint_indices_t.numel() == 0:
                continue
            active = mask[:, g_idx]  # (B,)
            # Expand to (B, 1, V, len(joints), 1) and apply to all T frames.
            active_t = active.view(B, 1, 1, 1, 1).expand(-1, T, V, len(joint_indices_t), 1)
            # Confidence channel is always zeroed (last channel).
            x[:, :, :, joint_indices_t, -1:] = torch.where(
                active_t,
                0.0,
                x[:, :, :, joint_indices_t, -1:],
            )
            if self.zero_coords:
                x[:, :, :, joint_indices_t, :2] = torch.where(
                    active_t.expand(-1, T, V, len(joint_indices_t), 2),
                    0.0,
                    x[:, :, :, joint_indices_t, :2],
                )
        return x

    def state_dict(self) -> dict:
        """Return serialisable state (useful for deterministic evals)."""
        return {
            "skeleton": self.skeleton,
            "group_rate": self.group_rate,
            "joint_rate": self.joint_rate,
            "temporal_consistency": self.temporal_consistency,
            "zero_coords": self.zero_coords,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: dict):
        self.skeleton = state["skeleton"]
        self.group_rate = state["group_rate"]
        self.joint_rate = state["joint_rate"]
        self.temporal_consistency = state["temporal_consistency"]
        self.zero_coords = state["zero_coords"]
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
