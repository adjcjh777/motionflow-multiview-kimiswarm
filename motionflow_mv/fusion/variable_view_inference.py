"""Variable view-count inference helpers for fixed-view fusion models.

A fixed-view ray-attention model is trained with a specific number of cameras
``n_views``.  At deployment we may have fewer views available, or we may want to
evaluate robustness when some views are dropped.  This module provides a
**zero-confidence masking** strategy that lets a fixed-view model run inference
with any subset of its expected views, without retraining or changing the
model.

The idea is simple: keep all ``n_views`` camera slots, but set the confidence of
any dropped view to zero.  The learned weight head still predicts weights for
those views, yet the subsequent triangulation multiplies weights by
confidences, so dropped views contribute nothing to the DLT solve.  Attention
layers still process the dropped views, but because the observations are
zero-confidence padding the model tends to learn to ignore them.

Future work: train a model with geometry-based camera positional encoding so
that views can be added/removed without fixed slots at all.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def apply_view_mask(
    x: torch.Tensor,
    active_views: torch.Tensor,
) -> torch.Tensor:
    """Zero out confidence channels of inactive views.

    Args:
        x: (..., V, J, 3) tensor of (x_pixel, y_pixel, confidence).
        active_views: (V,) bool tensor. ``True`` -> view is used.

    Returns:
        x masked: (..., V, J, 3) with confidence[..., inactive_views, :, 2] = 0.
    """
    if active_views.dtype != torch.bool:
        active_views = active_views.bool()
    x = x.clone()
    # Zero out both pixel and confidence for inactive views so the model has no
    # information from them.
    x[..., ~active_views, :, :] = 0.0
    return x


def prepare_variable_view_input(
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    active_views: Union[int, List[int], torch.Tensor],
    n_views_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a variable-view input so it can be fed to a fixed-view model.

    The returned ``x`` is padded/truncated to ``n_views_max`` and the confidence
    of every inactive view is set to zero.  The returned ``active_views`` mask
    has length ``n_views_max``.

    Args:
        x: (..., V_actual, J, 3) or (..., n_views_max, J, 3).
        K: (..., V_actual, 3, 3) or (..., n_views_max, 3, 3).
        R: (..., V_actual, 3, 3) or (..., n_views_max, 3, 3).
        t: (..., V_actual, 3) or (..., n_views_max, 3).
        active_views: either an int k (use first k views), a list of view
            indices, or a bool tensor of length V_actual or n_views_max.
        n_views_max: fixed view count expected by the model.  If None, infer
            from ``x``'s view dimension.

    Returns:
        x_padded, K_padded, R_padded, t_padded, active_views_mask
        where active_views_mask has length n_views_max.
    """
    if n_views_max is None:
        n_views_max = x.shape[-3]

    V_actual = x.shape[-3]

    if isinstance(active_views, int):
        k = min(active_views, n_views_max)
        mask = torch.zeros(n_views_max, dtype=torch.bool, device=x.device)
        mask[:k] = True
    elif isinstance(active_views, (list, tuple, np.ndarray)):
        mask = torch.zeros(n_views_max, dtype=torch.bool, device=x.device)
        indices = torch.as_tensor(active_views, device=x.device, dtype=torch.long)
        mask[indices] = True
    else:
        mask = active_views
        if mask.shape[0] < n_views_max:
            pad = torch.zeros(n_views_max - mask.shape[0], dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, pad], dim=0)
        mask = mask[:n_views_max]

    def _pad_to_view_axis(ten: torch.Tensor, view_axis: int) -> torch.Tensor:
        V = ten.shape[view_axis]
        if V == n_views_max:
            return ten
        if V < n_views_max:
            pad_shape = list(ten.shape)
            pad_shape[view_axis] = n_views_max - V
            pad = torch.zeros(pad_shape, device=ten.device, dtype=ten.dtype)
            return torch.cat([ten, pad], dim=view_axis)
        raise ValueError(f"Input has {V} views but n_views_max={n_views_max}")

    x = _pad_to_view_axis(x, view_axis=-3)
    K = _pad_to_view_axis(K, view_axis=-3)
    R = _pad_to_view_axis(R, view_axis=-3)
    t = _pad_to_view_axis(t, view_axis=-2)

    x = apply_view_mask(x, mask)
    return x, K, R, t, mask


def generate_view_subsets(
    n_views: int,
    min_views: int = 2,
    max_views: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Generate all subsets of views from min_views to max_views inclusive.

    Returns a list of tuples, one for each subset size, ordered by subset size.
    """
    if max_views is None:
        max_views = n_views
    subsets: List[Tuple[int, ...]] = []
    for k in range(min_views, max_views + 1):
        from itertools import combinations
        subsets.extend(combinations(range(n_views), k))
    return subsets


class VariableViewInferenceWrapper:
    """Wrap a fixed-view model so it can be evaluated with variable view counts.

    The wrapper does **not** modify the model parameters.  It simply masks out
    dropped views by zeroing their confidence before the forward pass.  This is
    compatible with any ``RayAttentionFusionModel*`` that follows the
    convention ``weights = sigmoid(logits) * confidences``.

    Example:
        >>> model = RayAttentionFusionModelTemporalResidual(n_views=4)
        >>> wrapper = VariableViewInferenceWrapper(model)
        >>> pred, weights = wrapper(x, K, R, t, active_views=[0, 2])
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def __call__(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        active_views: Union[int, List[int], torch.Tensor],
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, K, R, t, _ = prepare_variable_view_input(x, K, R, t, active_views)
        with torch.no_grad():
            return self.model(x, K=K, R=R, t=t, **model_kwargs)


def _make_cameras(n_views: int = 4):
    """Build a simple circular rig of pinhole cameras (helper for tests)."""
    from motionflow_mv.calibration.camera import Camera
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / np.linalg.norm(c)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t))
    return cameras


if __name__ == "__main__":
    # Smoke test: a 4-view model should run with 2, 3 or 4 active views.
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root))

    from motionflow_mv.calibration.camera import Camera
    from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual

    V, J, T = 4, 17, 5
    cameras = _make_cameras(V)
    x = torch.rand(2, T, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    model = RayAttentionFusionModelTemporalResidual(j=J, d=64, n_views=V)

    for k in [2, 3, 4]:
        wrapper = VariableViewInferenceWrapper(model)
        pred, w = wrapper(x, K, R, t, active_views=k)
        assert pred.shape == (2, T, J, 3), pred.shape
        assert w.shape[-1] == J and w.shape[-2] == V
        print(f"variable view inference smoke test passed for k={k}")
