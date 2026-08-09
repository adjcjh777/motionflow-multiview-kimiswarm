"""View-dropout augmentation for sparse-view generalization (v46).

The helper randomly drops camera views during training so the model learns to
work with sparse and variable camera configurations.  It preserves at least
``min_views`` views per batch element and returns a binary mask that downstream
modules (e.g. ``SparseViewGeneralizationV46``) can use to ignore dropped views.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _view_axis_dim(views: torch.Tensor) -> int:
    """Return the number of views along the canonical view axis.

    The view axis is the third-to-last dimension, i.e. ``views.shape[-3]``.
    """
    if views.dim() < 3:
        raise ValueError(
            f"views must have at least 3 dimensions, got shape {views.shape}"
        )
    return int(views.shape[-3])


def _broadcast_view_mask_to_views(
    views: torch.Tensor, view_mask: torch.Tensor
) -> torch.Tensor:
    """Broadcast a (B, V) view mask to the full shape of ``views``.

    Args:
        views: Tensor of shape ``(B, ..., V, J, C)`` or ``(B, V, J, C)``.
        view_mask: Tensor of shape ``(B, V)``.

    Returns:
        Mask broadcast to the same shape as ``views``.
    """
    B, V = view_mask.shape
    # Align V with the third-to-last axis of views.
    # For (B, T, V, J, C): (B, 1, V, 1, 1)
    # For (B, V, J, C)   : (B, V, 1, 1)
    n_middle = max(0, views.dim() - 4)
    return view_mask.view(B, *([1] * n_middle), V, 1, 1)


def drop_views(
    views: torch.Tensor,
    cameras: Optional[Dict[str, torch.Tensor]] = None,
    prob: float = 0.3,
    min_views: int = 2,
    curriculum_progress: Optional[float] = None,
    seed: Optional[int] = None,
    confidence_channel: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop camera views for a clip during training.

    Args:
        views: Input tensor with shape ``(B, ..., V, J, C)`` or
            ``(B, V, J, C)``.  The view axis is the third-to-last dimension.
        cameras: Optional dict with camera parameters (``K``, ``R``, ``t``).
            Kept for API compatibility with future camera-aware dropout;
            currently the cameras are returned unchanged.
        prob: Probability of dropping each view independently.
        min_views: Minimum number of views to keep per batch element.
        curriculum_progress: If given, scales ``prob`` from 0 at ``progress=0``
            to ``prob`` at ``progress=1``.  Useful for curriculum learning.
        seed: Optional random seed for reproducibility.
        confidence_channel: Channel index to zero out for dropped views.
            Set to ``None`` to leave all channels untouched and only return the
            mask.

    Returns:
        Tuple of ``(augmented_views, view_mask)``.  ``view_mask`` has shape
        ``(B, V)`` where ``1.0`` means the view is active and ``0.0`` means it
        is dropped.
    """
    # pylint: disable=unused-argument  # cameras reserved for future use
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if views.dim() < 4:
        raise ValueError(
            f"views must have at least 4 dimensions (B, ..., V, J, C), "
            f"got shape {views.shape}"
        )

    B = views.shape[0]
    V = _view_axis_dim(views)

    if min_views < 0:
        raise ValueError(f"min_views must be >= 0, got {min_views}")
    if min_views > V:
        raise ValueError(
            f"min_views ({min_views}) cannot exceed number of views ({V})"
        )

    if curriculum_progress is not None:
        p = prob * float(torch.clip(torch.tensor(curriculum_progress), 0.0, 1.0))
    else:
        p = prob

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=views.device)
        generator.manual_seed(seed)

    # Independent Bernoulli mask: 1.0 means keep.
    rand = torch.rand(B, V, device=views.device, generator=generator)
    view_mask = (rand >= p).float()

    # Enforce min_views per batch element by randomly promoting dropped views.
    if min_views > 0:
        for i in range(B):
            active = view_mask[i].nonzero(as_tuple=True)[0]
            if active.numel() < min_views:
                needed = min_views - active.numel()
                dropped = (view_mask[i] == 0).nonzero(as_tuple=True)[0]
                if dropped.numel() > 0:
                    perm = torch.randperm(
                        dropped.numel(), generator=generator
                    )
                    extra = dropped[perm[:needed]]
                    view_mask[i, extra] = 1.0

    # Apply the mask to the confidence channel so dropped views contribute
    # nothing to downstream triangulation/attention.
    views_aug = views.clone()
    if confidence_channel is not None:
        if not (0 <= confidence_channel < views.shape[-1]):
            raise ValueError(
                f"confidence_channel {confidence_channel} out of bounds for "
                f"last dim {views.shape[-1]}"
            )
        mask_expanded = _broadcast_view_mask_to_views(views, view_mask)
        views_aug[..., confidence_channel] = (
            views_aug[..., confidence_channel] * mask_expanded.squeeze(-1)
        )

    return views_aug, view_mask


class ViewDropoutAugmentationV46:
    """Stateful wrapper around :func:`drop_views`.

    Parameters
    ----------
    dropout_rate:
        Probability of dropping each view.
    min_views:
        Minimum number of views to keep per batch element.
    curriculum:
        If True, linearly ramps ``dropout_rate`` from 0 to ``dropout_rate``
        based on the progress passed to ``__call__``.
    seed:
        Optional random seed.  If None, a non-deterministic generator is used.
    confidence_channel:
        Channel index to zero out for dropped views.
    """

    def __init__(
        self,
        dropout_rate: float = 0.3,
        min_views: int = 2,
        curriculum: bool = True,
        seed: Optional[int] = None,
        confidence_channel: int = 2,
    ):
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )
        if min_views < 0:
            raise ValueError(f"min_views must be >= 0, got {min_views}")
        self.dropout_rate = dropout_rate
        self.min_views = min_views
        self.curriculum = curriculum
        self.confidence_channel = confidence_channel
        self._seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(
        self,
        views: torch.Tensor,
        cameras: Optional[Dict[str, torch.Tensor]] = None,
        progress: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply view dropout.

        Args:
            views: Input tensor of shape ``(B, ..., V, J, C)``.
            cameras: Optional camera dict (currently unused).
            progress: Training progress in ``[0, 1]``.  When ``curriculum`` is
                enabled, the effective dropout probability is
                ``dropout_rate * progress``.

        Returns:
            ``(augmented_views, view_mask)``.
        """
        effective_prob = self.dropout_rate
        if self.curriculum and progress is not None:
            effective_prob = self.dropout_rate * float(
                torch.clip(torch.tensor(progress), 0.0, 1.0)
            )
        return drop_views(
            views,
            cameras=cameras,
            prob=effective_prob,
            min_views=self.min_views,
            seed=self._seed,
            confidence_channel=self.confidence_channel,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dropout_rate": self.dropout_rate,
            "min_views": self.min_views,
            "curriculum": self.curriculum,
            "confidence_channel": self.confidence_channel,
            "seed": self._seed,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.dropout_rate = state["dropout_rate"]
        self.min_views = state["min_views"]
        self.curriculum = state["curriculum"]
        self.confidence_channel = state["confidence_channel"]
        self._seed = state["seed"]
        self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))


if __name__ == "__main__":
    # Smoke test for the view-dropout augmentation helper.
    B, T, V, J, C = 2, 5, 4, 17, 3
    views = torch.rand(B, T, V, J, C)
    views[..., 2] = torch.rand(B, T, V, J)

    # Basic drop.
    aug, mask = drop_views(views, prob=0.3, min_views=2)
    assert aug.shape == views.shape
    assert mask.shape == (B, V)
    assert mask.dtype == torch.float32
    assert mask.sum(dim=1).ge(2).all()

    # Dropped views have zero confidence.
    for b in range(B):
        for v in range(V):
            if mask[b, v].item() == 0.0:
                assert aug[b, :, v, :, 2].abs().max().item() < 1e-6

    # Determinism with seed.
    aug1, mask1 = drop_views(views, prob=0.5, seed=42)
    aug2, mask2 = drop_views(views, prob=0.5, seed=42)
    assert torch.equal(aug1, aug2)
    assert torch.equal(mask1, mask2)

    # Wrapper with curriculum.
    wrapper = ViewDropoutAugmentationV46(
        dropout_rate=0.5, min_views=2, curriculum=True, seed=123
    )
    aug_w, mask_w = wrapper(views, progress=0.5)
    assert aug_w.shape == views.shape
    assert mask_w.shape == (B, V)
    assert mask_w.sum(dim=1).ge(2).all()

    # Edge case: prob=0 keeps all views.
    aug_full, mask_full = drop_views(views, prob=0.0, min_views=2)
    assert torch.equal(mask_full, torch.ones_like(mask_full))

    print("view_dropout_augmentation_v46 smoke tests passed")
