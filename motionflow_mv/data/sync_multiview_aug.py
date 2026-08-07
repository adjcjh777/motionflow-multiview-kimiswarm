"""Synchronized multi-view 2D keypoint augmentation.

In a multi-view setup, applying different 2D geometric augmentations to
different camera views breaks epipolar consistency.  This module applies the
*same* image-space transformation to every view for a given sample, so the
underlying multi-view geometry remains valid.

Expected input layout: ``(..., V, J, C)`` where the last dimension contains at
least two coordinate channels (x, y).  Only the x, y channels are modified;
confidence / visibility channels are left unchanged.

API summary
-----------
* :func:`flip_horizontal`      -- horizontal flip around a vertical axis.
* :func:`rotate`               -- rotation around a center point.
* :func:`scale`                -- uniform scaling around a center point.
* :func:`translate`            -- pixel / normalized translation.
* :class:`SynchronizedMultiview2DAugmenter` -- configurable, stateful augmenter.
"""

import math
from typing import Optional, Tuple, Union

import torch

Number = Union[int, float]


def _resolve_center(
    image_size: Optional[Tuple[int, int]], default: Tuple[float, float]
) -> Tuple[float, float]:
    """Resolve the 2D center used by geometric transforms.

    Args:
        image_size: Optional ``(W, H)`` image size.  If provided, the center is
            ``(W / 2, H / 2)``.
        default: Fallback center ``(cx, cy)`` when ``image_size`` is ``None``.

    Returns:
        ``(cx, cy)`` center coordinates.
    """
    if image_size is not None:
        return image_size[0] / 2.0, image_size[1] / 2.0
    return default


def flip_horizontal(
    x: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Flip 2D keypoints horizontally around a vertical axis.

    The same axis is used for every view, preserving multi-view consistency.

    Args:
        x: Input tensor of shape ``(..., V, J, C)`` with ``C >= 2``.
        image_size: Optional ``(W, H)`` image size.  If provided, the flip axis
            is ``W / 2``.
        center: Optional ``(cx, cy)`` override.  If ``image_size`` is also given,
            ``image_size`` takes precedence.

    Returns:
        Flipped tensor with the same shape as ``x``.
    """
    if x.dim() < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions, got shape {tuple(x.shape)}"
        )
    if image_size is not None:
        cx = image_size[0] / 2.0
    elif center is not None:
        cx = center[0]
    else:
        cx = 0.0
    x = x.clone()
    x[..., 0] = 2.0 * cx - x[..., 0]
    return x


def rotate(
    x: torch.Tensor,
    angle_deg: Number,
    image_size: Optional[Tuple[int, int]] = None,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Rotate 2D keypoints around a center point.

    Args:
        x: Input tensor of shape ``(..., V, J, C)`` with ``C >= 2``.
        angle_deg: Rotation angle in degrees.  Positive values rotate
            counter-clockwise.
        image_size: Optional ``(W, H)`` image size.  If provided, the rotation
            center is the image center.
        center: Optional ``(cx, cy)`` override when ``image_size`` is not given.

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    if x.dim() < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions, got shape {tuple(x.shape)}"
        )
    cx, cy = _resolve_center(image_size, center or (0.0, 0.0))
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x = x.clone()
    dx = x[..., 0] - cx
    dy = x[..., 1] - cy
    x[..., 0] = cx + cos_t * dx - sin_t * dy
    x[..., 1] = cy + sin_t * dx + cos_t * dy
    return x


def scale(
    x: torch.Tensor,
    scale_factor: Number,
    image_size: Optional[Tuple[int, int]] = None,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Uniformly scale 2D keypoints around a center point.

    Args:
        x: Input tensor of shape ``(..., V, J, C)`` with ``C >= 2``.
        scale_factor: Isotropic scale factor.
        image_size: Optional ``(W, H)`` image size.  If provided, the scaling
            center is the image center.
        center: Optional ``(cx, cy)`` override when ``image_size`` is not given.

    Returns:
        Scaled tensor with the same shape as ``x``.
    """
    if x.dim() < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions, got shape {tuple(x.shape)}"
        )
    cx, cy = _resolve_center(image_size, center or (0.0, 0.0))
    s = float(scale_factor)
    x = x.clone()
    x[..., 0] = cx + s * (x[..., 0] - cx)
    x[..., 1] = cy + s * (x[..., 1] - cy)
    return x


def translate(
    x: torch.Tensor,
    dx: Number,
    dy: Number,
) -> torch.Tensor:
    """Translate 2D keypoints by ``(dx, dy)``.

    Args:
        x: Input tensor of shape ``(..., V, J, C)`` with ``C >= 2``.
        dx: Horizontal translation.
        dy: Vertical translation.

    Returns:
        Translated tensor with the same shape as ``x``.
    """
    if x.dim() < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions, got shape {tuple(x.shape)}"
        )
    x = x.clone()
    x[..., 0] = x[..., 0] + float(dx)
    x[..., 1] = x[..., 1] + float(dy)
    return x


def _sample_scalar(
    shape: Tuple[int, ...],
    low: Number,
    high: Number,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a tensor of uniform scalar values with the given leading shape."""
    if low == high:
        return torch.full(shape, float(low), dtype=torch.float32)
    device = generator.device if generator is not None else "cpu"
    values = (
        torch.rand(shape, generator=generator, device=device)
        * (float(high) - float(low))
        + float(low)
    )
    return values


class SynchronizedMultiview2DAugmenter:
    """Configurable synchronized multi-view 2D augmenter.

    For each sample, a single set of transform parameters is drawn and applied
    identically to all views, preserving multi-view epipolar consistency.

    Args:
        horizontal_flip_prob: Probability of applying a horizontal flip.
        rotation_deg: Maximum rotation in degrees.  Rotation angles are sampled
            uniformly from ``[-rotation_deg, rotation_deg]``.  Set to ``0`` to
            disable.
        scale_range: ``(min, max)`` uniform scale factors.  Set to ``(1, 1)``
            to disable.
        translation_px: Maximum translation in pixels per axis.  The actual
            offset is sampled uniformly from ``[-translation_px, translation_px]``.
            Set to ``0`` to disable.
        image_size: ``(W, H)`` image size used to compute the center for flip,
            rotation and scaling.  If ``None``, the origin ``(0, 0)`` is used as
            the center.
        per_sample: If ``True``, sample transform parameters independently for
            each leading sample (batch / clip).  If ``False``, use a single set
            of parameters for the whole tensor.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        rotation_deg: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_px: float = 5.0,
        image_size: Optional[Tuple[int, int]] = None,
        per_sample: bool = True,
        seed: Optional[int] = None,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.translation_px = translation_px
        self.image_size = image_size
        self.per_sample = per_sample
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def state_dict(self) -> dict:
        """Return a serialisable state (useful for deterministic evals)."""
        return {
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "rotation_deg": self.rotation_deg,
            "scale_range": self.scale_range,
            "translation_px": self.translation_px,
            "image_size": self.image_size,
            "per_sample": self.per_sample,
            "generator_state": self.generator.get_state().tolist(),
        }

    def load_state_dict(self, state: dict):
        self.horizontal_flip_prob = state["horizontal_flip_prob"]
        self.rotation_deg = state["rotation_deg"]
        self.scale_range = tuple(state["scale_range"])
        self.translation_px = state["translation_px"]
        self.image_size = (
            tuple(state["image_size"])
            if state["image_size"] is not None
            else None
        )
        self.per_sample = state["per_sample"]
        self.generator.set_state(
            torch.tensor(state["generator_state"], dtype=torch.uint8)
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply synchronized 2D augmentation to ``x``.

        Args:
            x: Input tensor of shape ``(..., V, J, C)`` with ``C >= 2``.

        Returns:
            Augmented tensor with the same shape as ``x``; the input is left
            unchanged.
        """
        if x.dim() < 3:
            raise ValueError(
                f"Input must have at least 3 dimensions, got shape {tuple(x.shape)}"
            )

        original_shape = tuple(x.shape)
        x = x.clone()

        if self.per_sample:
            n_leading = (
                int(torch.prod(torch.tensor(original_shape[:-3], dtype=torch.int64)))
                if x.dim() > 3
                else 1
            )
            x_view = x.view(n_leading, *original_shape[-3:])
        else:
            x_view = x.unsqueeze(0)
            n_leading = 1

        # Sample per-sample parameters.
        param_shape = (n_leading,)
        flip_flags = _sample_scalar(
            param_shape, 0.0, 1.0, self.generator
        ) < self.horizontal_flip_prob
        angles = _sample_scalar(
            param_shape, -self.rotation_deg, self.rotation_deg, self.generator
        )
        scale_factors = _sample_scalar(
            param_shape, self.scale_range[0], self.scale_range[1], self.generator
        )
        dx = _sample_scalar(
            param_shape,
            -self.translation_px,
            self.translation_px,
            self.generator,
        )
        dy = _sample_scalar(
            param_shape,
            -self.translation_px,
            self.translation_px,
            self.generator,
        )

        # Apply transforms sequentially per sample so the same transformation is
        # applied to every view.
        out = []
        for i in range(n_leading):
            sample = x_view[i : i + 1]

            if flip_flags[i].item():
                sample = flip_horizontal(sample, image_size=self.image_size)
            if angles[i].item() != 0.0:
                sample = rotate(
                    sample, angle_deg=angles[i].item(), image_size=self.image_size
                )
            sample = scale(
                sample, scale_factor=scale_factors[i].item(), image_size=self.image_size
            )
            sample = translate(sample, dx=dx[i].item(), dy=dy[i].item())
            out.append(sample)

        x_aug = torch.cat(out, dim=0).view(original_shape)
        return x_aug
