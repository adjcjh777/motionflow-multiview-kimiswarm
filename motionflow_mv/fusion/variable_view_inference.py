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


def _fill_camera_params(
    ten: torch.Tensor,
    active_views: torch.Tensor,
    view_axis: int,
    fill_mode: str = "last_active",
) -> torch.Tensor:
    """Fill inactive camera slots with valid parameters instead of zeros.

    Args:
        ten: tensor with a view axis of length ``n_views_max``.
        active_views: bool tensor of length ``n_views_max``.
        view_axis: axis that corresponds to views.
        fill_mode: ``"last_active"`` copies the last active view into every
            inactive slot; ``"mean_active"`` uses the mean of active views.

    Returns:
        Tensor of the same shape with inactive slots filled.
    """
    if fill_mode not in ("last_active", "mean_active"):
        return ten

    active_indices = torch.where(active_views)[0]
    if active_indices.numel() == 0:
        return ten

    # Work on a copy and move the view axis to position 0 for vectorisation.
    out = ten.clone()
    view_axis_int = view_axis if view_axis >= 0 else ten.dim() + view_axis
    perm = list(range(ten.dim()))
    perm.insert(0, perm.pop(view_axis_int))
    out = out.permute(*perm).contiguous()

    inactive_indices = torch.where(~active_views)[0]
    if fill_mode == "last_active":
        fill_value = out[active_indices[-1].item()]
    else:  # mean_active
        fill_value = out[active_indices].mean(dim=0, keepdim=True)

    for idx in inactive_indices:
        out[idx] = fill_value

    # Inverse permutation.
    inv_perm = [0] * ten.dim()
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return out.permute(*inv_perm).contiguous()


def prepare_variable_view_input(
    x: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    active_views: Union[int, List[int], torch.Tensor],
    n_views_max: Optional[int] = None,
    fill_camera_mode: str = "last_active",
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
        fill_camera_mode: How to fill camera parameters for padded/inactive view
            slots.  ``"last_active"`` copies the last active view, avoiding
            zero intrinsics/extrinsics that can destabilise ray embedding and
            triangulation.  Use ``"zero"`` for the legacy zero-padding
            behaviour.

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

    # Replace zero-padded camera parameters with valid values so the model has
    # well-defined intrinsics/extrinsics for inactive views, even though their
    # observations are masked out.
    if fill_camera_mode in ("last_active", "mean_active"):
        K = _fill_camera_params(K, mask, view_axis=-3, fill_mode=fill_camera_mode)
        R = _fill_camera_params(R, mask, view_axis=-3, fill_mode=fill_camera_mode)
        t = _fill_camera_params(t, mask, view_axis=-2, fill_mode=fill_camera_mode)

    x = apply_view_mask(x, mask)
    return x, K, R, t, mask


def build_active_view_edge_index(
    j: int,
    parents: List[int],
    symmetry_pairs: List[Tuple[int, int]],
    active_views: Union[int, List[int], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a (view, joint) edge index restricted to active views.

    The returned edge index uses the original view indices in the full
    ``n_views_max`` rig, so it can be passed directly to
    :class:`GraphJointAttentionV2` without changing its ``n_views`` attribute.

    Args:
        j: Number of joints.
        parents: Skeleton parent list, ``-1`` for root.
        symmetry_pairs: List of symmetric joint pairs.
        active_views: int, list of indices, or bool tensor indicating which
            views are active.

    Returns:
        edge_index, edge_type tensors over only the active views.
    """
    if isinstance(active_views, int):
        active_indices = list(range(active_views))
    elif isinstance(active_views, (list, tuple, np.ndarray)):
        active_indices = list(active_views)
    else:
        mask = active_views.bool()
        active_indices = torch.where(mask)[0].tolist()

    n_active = len(active_indices)
    if n_active == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )

    # Build edge index for a reduced rig and map view indices back.
    from motionflow_mv.fusion.graph_joint_attention_v2 import build_graph_joint_edge_index

    edge_index, edge_type = build_graph_joint_edge_index(
        parents, symmetry_pairs, n_active, j, add_self_loops=True
    )
    device = edge_index.device

    # edge_index contains node ids = local_view * j + joint.  Map local_view to
    # the real active view index.
    view_idx = edge_index // j
    joint_idx = edge_index % j
    index_map = torch.tensor(active_indices, dtype=torch.long, device=device)
    mapped_view_idx = index_map[view_idx]
    mapped_edge_index = mapped_view_idx * j + joint_idx
    return mapped_edge_index, edge_type


def set_graph_joint_attention_active_views(
    model: nn.Module,
    j: int,
    active_views: Union[int, List[int], torch.Tensor],
) -> None:
    """Restrict a model's graph-joint attention graph to active views.

    The function infers the skeleton from ``j`` (17 -> H36M, 28 -> MPI-INF-3DHP)
    and overwrites ``model.graph_joint_attention.edge_index`` and
    ``edge_type`` with the subgraph.  If the model has no graph-joint attention
    block, this is a no-op.

    Args:
        model: Fusion model, potentially with a ``graph_joint_attention`` block.
        j: Number of joints in the current skeleton.
        active_views: active view specifier.
    """
    from motionflow_mv.fusion.graph_joint_relation import (
        H36M_17_PARENTS,
        H36M_17_SYMMETRY_PAIRS,
        MPI_INF_3DHP_28_PARENTS,
        MPI_INF_3DHP_28_SYMMETRY_PAIRS,
    )

    if not hasattr(model, "graph_joint_attention") or model.graph_joint_attention is None:
        return

    if j == 17:
        parents = H36M_17_PARENTS
        symmetry = H36M_17_SYMMETRY_PAIRS
    elif j == 28:
        parents = MPI_INF_3DHP_28_PARENTS
        symmetry = MPI_INF_3DHP_28_SYMMETRY_PAIRS
    else:
        # Default to H36M skeleton; caller can rebuild explicitly if needed.
        parents = H36M_17_PARENTS
        symmetry = H36M_17_SYMMETRY_PAIRS

    edge_index, edge_type = build_active_view_edge_index(j, parents, symmetry, active_views)
    model.graph_joint_attention.edge_index = edge_index
    model.graph_joint_attention.edge_type = edge_type


def confidence_fallback_triangulate(
    points_2d: torch.Tensor,
    P: torch.Tensor,
    weights: Optional[torch.Tensor],
    active_views: torch.Tensor,
    min_views: int = 2,
) -> Optional[torch.Tensor]:
    """Return a direct DLT estimate when too few views are active.

    Uses only the active views and their confidence-weighted observations.  If
    the number of active views is at least ``min_views``, returns ``None`` so
    the caller can use the model's own output.

    Args:
        points_2d: (N, V, J, 2) image points.
        P: (N, V, 3, 4) projection matrices.
        weights: (N, V, J) per-view per-joint weights, or None.
        active_views: bool tensor of length V.
        min_views: minimum number of views before the fallback is used.

    Returns:
        (N, J, 3) triangulated points if active views < min_views, else None.
    """
    active_indices = torch.where(active_views)[0]
    if active_indices.numel() >= min_views:
        return None
    if active_indices.numel() < 2:
        # Degenerate: fewer than 2 views cannot triangulate.  Return zeros.
        N, _, J, _ = points_2d.shape
        return torch.zeros((N, J, 3), dtype=points_2d.dtype, device=points_2d.device)

    from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

    points_active = points_2d[:, active_indices]
    P_active = P[:, active_indices]
    weights_active = weights[:, active_indices] if weights is not None else None
    return triangulate_dlt_batched_lstsq(points_active, P_active, weights_active)


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


class HardenedVariableViewInferenceWrapper(VariableViewInferenceWrapper):
    """Variable-view wrapper with explicit hardening for few-view inference.

    Compared with :class:`VariableViewInferenceWrapper`, this wrapper:

    1. Fills inactive camera slots with valid parameters instead of zeros.
    2. Restricts graph-joint attention to active views when the model exposes a
       ``graph_joint_attention`` block.
    3. Falls back to plain confidence-weighted DLT when the number of active
       views is below ``min_views``.

    These mitigations address the catastrophic 2/3-view failure mode observed
    when a 4-view model is evaluated with fewer cameras.

    Example:
        >>> model = OmniMultiViewFusionV2(j=17, n_views=4)
        >>> wrapper = HardenedVariableViewInferenceWrapper(model, min_views=2)
        >>> pred, weights, visibility, cov, epi = wrapper(x, K, R, t, active_views=[0, 2])
    """

    def __init__(
        self,
        model: nn.Module,
        min_views: int = 2,
        fill_camera_mode: str = "last_active",
    ):
        super().__init__(model)
        self.min_views = max(2, min_views)
        self.fill_camera_mode = fill_camera_mode

    def __call__(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        active_views: Union[int, List[int], torch.Tensor],
        **model_kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        x_in = x
        K_in = K
        R_in = R
        t_in = t

        x, K, R, t, mask = prepare_variable_view_input(
            x, K, R, t, active_views, fill_camera_mode=self.fill_camera_mode
        )

        # Restrict graph-joint attention to active views, if present.
        B, T, V, J, _ = x.shape
        set_graph_joint_attention_active_views(self.model, J, mask)

        with torch.no_grad():
            outputs = self.model(x, K=K, R=R, t=t, **model_kwargs)

        pred_3d = outputs[0]
        weights = outputs[1]

        # If active views < min_views, replace the model output with direct DLT.
        n_active = int(mask.sum().item())
        if n_active < self.min_views:
            # Prepare projection matrices from the padded camera parameters.
            Rt = torch.cat([R, t[..., None]], dim=-1)  # (B*T, V, 3, 4)
            P = K @ Rt  # (B*T, V, 3, 4)

            from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

            # Use the original 2D points and confidences; inactive views have
            # zero confidence so they do not affect the DLT solve.
            x_flat = x.reshape(B * T, V, J, 3)
            points_2d = x_flat[..., :2]
            confidences = x_flat[..., 2]

            fallback = triangulate_dlt_batched_lstsq(points_2d, P, confidences)

            # Reshape back to the model's output shape.
            if pred_3d.dim() == 3:  # (B, J, 3) single-frame
                pred_3d = fallback.view(B, J, 3)
            else:
                pred_3d = fallback.view(B, T, J, 3)

            # Re-pack outputs with the fallback prediction.
            outputs = (pred_3d,) + outputs[1:]

        return outputs


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
    from motionflow_mv.fusion.omniview_fusion_v2 import OmniMultiViewFusionV2
    from motionflow_mv.fusion.graph_joint_relation import (
        H36M_17_PARENTS,
        H36M_17_SYMMETRY_PAIRS,
    )

    V, J, T = 4, 17, 5
    cameras = _make_cameras(V)
    x = torch.rand(2, T, V, J, 3)
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float()
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float()
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float()

    # Basic wrapper smoke test on a model without graph-joint attention.
    model = RayAttentionFusionModelTemporalResidual(j=J, d=64, n_views=V)

    for k in [2, 3, 4]:
        wrapper = VariableViewInferenceWrapper(model)
        pred, w = wrapper(x, K, R, t, active_views=k)
        assert pred.shape == (2, T, J, 3), pred.shape
        assert w.shape[-1] == J and w.shape[-2] == V
        print(f"VariableViewInferenceWrapper smoke test passed for k={k}")

    # Hardened wrapper smoke test on a model with graph-joint attention.
    model_v2 = OmniMultiViewFusionV2(j=J, d=32, n_views=V, graph_num_layers=1)
    hardened = HardenedVariableViewInferenceWrapper(model_v2, min_views=3)
    for k in [2, 3, 4]:
        pred, weights, visibility, cov, epi = hardened(x, K, R, t, active_views=k)
        assert pred.shape == (2, T, J, 3), pred.shape
        assert weights.shape[-1] == J and weights.shape[-2] == V
        print(f"HardenedVariableViewInferenceWrapper smoke test passed for k={k}")

    # Direct helper smoke tests.
    mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    edge_index, edge_type = build_active_view_edge_index(J, H36M_17_PARENTS, H36M_17_SYMMETRY_PAIRS, mask)
    assert edge_index.shape[1] > 0
    print("build_active_view_edge_index smoke test passed")
