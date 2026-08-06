"""Occlusion augmentation for multi-view 2D keypoint tensors.

The canonical input is a tensor of shape ``(..., V, J, C)`` where ``V`` is the
number of views, ``J`` the number of joints, and the last channel is assumed to
contain a confidence / visibility score (default channel index ``-1``).  All
occlusion operations work by zeroing this confidence channel; optionally the
2D coordinate channels can be zeroed as well.

API summary
-----------
* :func:`occlude_views`            -- drop whole camera views.
* :func:`occlude_joints`           -- drop individual joints.
* :func:`random_occlude_views`     -- randomly drop views according to a rate.
* :func:`random_occlude_joints`    -- randomly drop joints according to a rate.
* :class:`OcclusionAugmenter`      -- configurable, stateful augmenter.
"""

from typing import List, Optional, Tuple, Union

import torch

Number = Union[int, float]


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


def occlude_views(
    x: torch.Tensor,
    view_indices: Union[int, List[int], torch.Tensor],
    confidence_channel: int = -1,
    zero_coords: bool = False,
) -> torch.Tensor:
    """Occlude one or more entire camera views.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        view_indices: View index or list of view indices to occlude.
        confidence_channel: Index of the confidence/visibility channel. Default ``-1``.
        zero_coords: If ``True``, also zero the 2D coordinate channels.

    Returns:
        Tensor with the same shape as ``x``; confidences for the selected views
        are set to zero.
    """
    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    if isinstance(view_indices, int):
        view_indices = [view_indices]
    view_indices = torch.as_tensor(view_indices, dtype=torch.long, device=x.device)
    view_indices = view_indices[(view_indices >= 0) & (view_indices < V)]
    if view_indices.numel() == 0:
        return x

    # Flatten all leading dimensions so we can index (N, V, J, C).
    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])
    x_view[:, view_indices, :, c] = 0.0
    if zero_coords:
        x_view[:, view_indices, :, :2] = 0.0
    return x


def occlude_joints(
    x: torch.Tensor,
    joint_indices: Union[int, List[int], torch.Tensor],
    confidence_channel: int = -1,
    zero_coords: bool = False,
) -> torch.Tensor:
    """Occlude one or more joints across all views.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        joint_indices: Joint index or list of joint indices to occlude.
        confidence_channel: Index of the confidence/visibility channel. Default ``-1``.
        zero_coords: If ``True``, also zero the 2D coordinate channels.

    Returns:
        Tensor with the same shape as ``x``; confidences for the selected joints
        are set to zero in every view.
    """
    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    if isinstance(joint_indices, int):
        joint_indices = [joint_indices]
    joint_indices = torch.as_tensor(joint_indices, dtype=torch.long, device=x.device)
    joint_indices = joint_indices[(joint_indices >= 0) & (joint_indices < J)]
    if joint_indices.numel() == 0:
        return x

    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])
    x_view[:, :, joint_indices, c] = 0.0
    if zero_coords:
        x_view[:, :, joint_indices, :2] = 0.0
    return x


def random_occlude_views(
    x: torch.Tensor,
    rate: float,
    per_sample: bool = False,
    confidence_channel: int = -1,
    zero_coords: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly occlude camera views with probability ``rate``.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        rate: Probability of occluding each view (0 <= rate <= 1).
        per_sample: If ``True``, sample independently for each leading sample
            (e.g. batch element + frame). If ``False``, use a single mask for
            all leading samples.
        confidence_channel: Index of the confidence/visibility channel. Default ``-1``.
        zero_coords: If ``True``, also zero the 2D coordinate channels.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Occluded tensor.
    """
    if rate <= 0.0:
        return x
    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])

    if per_sample:
        mask = torch.rand(N, V, generator=generator, device=x.device) < rate
        x_view[mask, :, :, c] = 0.0
        if zero_coords:
            x_view[mask, :, :, :2] = 0.0
    else:
        mask = torch.rand(V, generator=generator, device=x.device) < rate
        x_view[:, mask, :, c] = 0.0
        if zero_coords:
            x_view[:, mask, :, :2] = 0.0
    return x


def random_occlude_joints(
    x: torch.Tensor,
    rate: float,
    per_view: bool = True,
    per_sample: bool = False,
    confidence_channel: int = -1,
    zero_coords: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly occlude joints with probability ``rate``.

    Args:
        x: Input tensor of shape ``(..., V, J, C)``.
        rate: Probability of occluding each joint (0 <= rate <= 1).
        per_view: If ``True``, sample joints independently per view. If
            ``False``, the same set of joints is occluded across all views for
            each sample.
        per_sample: If ``True``, sample independently for each leading sample.
            If ``False``, use a single mask shared across all leading samples.
        confidence_channel: Index of the confidence/visibility channel. Default ``-1``.
        zero_coords: If ``True``, also zero the 2D coordinate channels.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        Occluded tensor.
    """
    if rate <= 0.0:
        return x
    x = x.clone()
    V, J = _validate_view_joint_dims(x)
    c = _resolve_confidence_channel(x, confidence_channel)

    N = x.shape[:-3].numel() if x.dim() > 3 else 1
    x_view = x.view(N, V, J, x.shape[-1])

    if per_view:
        if per_sample:
            mask = torch.rand(N, V, J, generator=generator, device=x.device) < rate
        else:
            mask = torch.rand(V, J, generator=generator, device=x.device) < rate
    else:
        if per_sample:
            mask = torch.rand(N, J, generator=generator, device=x.device) < rate
            mask = mask.unsqueeze(2).expand(-1, -1, V)  # (N, J, V)
            mask = mask.permute(0, 2, 1)  # (N, V, J)
        else:
            mask = torch.rand(J, generator=generator, device=x.device) < rate
            mask = mask.view(1, 1, J).expand(N, V, -1)

    x_view[..., c] = x_view[..., c] * (~mask).float()
    if zero_coords:
        x_view[..., :2] = x_view[..., :2] * (~mask)[..., None].float()
    return x


class OcclusionAugmenter:
    """Convenience wrapper for random view + joint occlusion.

    Args:
        view_rate: Probability of occluding each entire view.
        joint_rate: Probability of occluding each (view, joint) detection.
        per_view: Whether joint occlusion is sampled independently per view.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        view_rate: float = 0.0,
        joint_rate: float = 0.0,
        per_view: bool = True,
        seed: Optional[int] = None,
    ):
        self.view_rate = view_rate
        self.joint_rate = joint_rate
        self.per_view = per_view
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply view occlusion then joint occlusion to ``x``.

        The operation is performed on a clone, so the input is left unchanged.
        """
        x = random_occlude_views(
            x,
            self.view_rate,
            per_sample=False,
            generator=self.generator,
        )
        x = random_occlude_joints(
            x,
            self.joint_rate,
            per_view=self.per_view,
            per_sample=False,
            generator=self.generator,
        )
        return x

    def state_dict(self) -> dict:
        """Return serialisable state (useful for deterministic evals)."""
        return {
            "view_rate": self.view_rate,
            "joint_rate": self.joint_rate,
            "per_view": self.per_view,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: dict):
        self.view_rate = state["view_rate"]
        self.joint_rate = state["joint_rate"]
        self.per_view = state["per_view"]
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))
