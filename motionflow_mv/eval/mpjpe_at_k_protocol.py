"""Standardised MPJPE@k evaluation protocol for variable-view multi-view pose.

The protocol answers: *given a fixed-view fusion model and a scene with V
views, what is the reconstruction error if only k views are available?*  For
every requested ``k`` we sample view subsets, run the model with only those
views active, and report MPJPE/PA-MPJPE/root-relative-MPJPE plus a temporal
jerk metric.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from motionflow_mv.eval.metrics import (
    mpjpe as mpjpe_metric,
    pa_mpjpe,
    root_relative_mpjpe,
)
from motionflow_mv.fusion.variable_view_inference import (
    HardenedVariableViewInferenceWrapper,
    VariableViewInferenceWrapper,
    prepare_variable_view_input,
)


# Valid alignment modes.  Keep the strings short so JSON outputs stay tidy.
_ALIGN_MODES = {"none", "pa", "root"}


def temporal_jerk(poses_m: np.ndarray) -> float:
    """Mean magnitude of the 3rd temporal derivative of ``poses_m`` (mm).

    Args:
        poses_m: (T, J, 3) array of 3D poses in metres.

    Returns:
        Scalar jerk in mm.  Returns 0.0 when ``T < 4``.
    """
    if poses_m.shape[0] < 4:
        return 0.0
    third_diff = np.diff(poses_m, n=3, axis=0)  # (T-3, J, 3) in metres
    return float(np.linalg.norm(third_diff, axis=-1).mean() * 1000.0)


def _align(pred: np.ndarray, gt: np.ndarray, align: str) -> np.ndarray:
    """Return a copy of ``pred`` optionally aligned to ``gt``.

    Args:
        pred: (T, J, 3) predictions in any unit.
        gt: (T, J, 3) ground truth in the same unit.
        align: one of ``"none"``, ``"pa"`` (Procrustes), ``"root"`` (pelvis-centred).

    Returns:
        Aligned predictions.
    """
    if align == "none":
        return pred.copy()
    if align == "pa":
        return pa_mpjpe_aligned(pred, gt)
    if align == "root":
        root_idx = 0
        return pred - pred[..., root_idx : root_idx + 1, :]
    raise ValueError(f"Unknown align mode: {align!r}. Use one of {_ALIGN_MODES}")


def pa_mpjpe_aligned(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-align each frame of ``pred`` to ``gt``.

    This is a thin wrapper around :func:`motionflow_mv.eval.metrics.pa_mpjpe`
    that returns the *aligned* prediction array instead of the scalar error.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    # metrics.py implements per-frame rigid alignment in _align_rigid_batch.
    from motionflow_mv.eval.metrics import _align_rigid_batch

    return _align_rigid_batch(pred, gt)


def compute_mpjpe_at_k(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    k: Optional[int] = None,
    align: str = "none",
    root_idx: int = 0,
) -> Dict[str, float]:
    """Compute MPJPE@k metrics for a single view-subset evaluation.

    Args:
        pred: (T, J, 3) predicted 3D poses in metres.
        gt: (T, J, 3) ground-truth 3D poses in metres.
        k: Optional view count label, stored in the returned dict for bookkeeping.
        align: ``"none"`` for plain MPJPE, ``"pa"`` for Procrustes-aligned,
            ``"root"`` for root-relative.
        root_idx: Root joint index for ``align="root"``.

    Returns:
        Dict with keys ``mpjpe``, ``pa_mpjpe`` (or root-relative), ``temporal_jerk``,
        and ``k``.
    """
    if align not in _ALIGN_MODES:
        raise ValueError(f"Unknown align={align!r}. Use one of {_ALIGN_MODES}")

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if align == "root":
        pred_rel = pred - pred[..., root_idx : root_idx + 1, :]
        gt_rel = gt - gt[..., root_idx : root_idx + 1, :]
        aligned_pred = pred_rel
        # For a fair root-relative metric, also centre gt.
        _ = gt_rel
        mpjpe_val = mpjpe_metric(pred_rel * 1000.0, gt_rel * 1000.0)
        pa_val = pa_mpjpe(pred_rel, gt_rel)
    elif align == "pa":
        aligned_pred = pa_mpjpe_aligned(pred, gt)
        mpjpe_val = mpjpe_metric(aligned_pred * 1000.0, gt * 1000.0)
        pa_val = pa_mpjpe(pred, gt)
    else:
        aligned_pred = pred
        mpjpe_val = mpjpe_metric(pred * 1000.0, gt * 1000.0)
        pa_val = pa_mpjpe(pred, gt)

    return {
        "k": k if k is not None else -1,
        "mpjpe": float(mpjpe_val),
        "pa_mpjpe": float(pa_val),
        "root_rel_mpjpe": float(root_relative_mpjpe(pred, gt, root_idx=root_idx)),
        "temporal_jerk": temporal_jerk(aligned_pred),
    }


def generate_view_subsets(
    V: int,
    k_values: Iterable[int],
    num_subsets_per_k: Optional[int] = None,
    seed: int = 42,
) -> Dict[int, List[Tuple[int, ...]]]:
    """Generate deterministic view subsets for each requested ``k``.

    Args:
        V: Total number of views.
        k_values: Requested view counts.
        num_subsets_per_k: If ``None`` enumerate all ``C(V, k)`` subsets; otherwise
            randomly sample this many without replacement.
        seed: RNG seed used only when sampling.

    Returns:
        Mapping ``k -> list of subsets`` where each subset is a sorted tuple of
        view indices.
    """
    rng = np.random.default_rng(seed)
    subsets_by_k: Dict[int, List[Tuple[int, ...]]] = {}
    for k in sorted(set(k_values)):
        if k < 1 or k > V:
            continue
        total = math.comb(V, k)
        if num_subsets_per_k is None or total <= num_subsets_per_k:
            subsets = [tuple(sorted(c)) for c in combinations(range(V), k)]
        else:
            seen: set = set()
            while len(seen) < num_subsets_per_k:
                subset = tuple(sorted(rng.choice(V, size=k, replace=False)))
                seen.add(subset)
            subsets = list(seen)
        subsets_by_k[k] = subsets
    return subsets_by_k


def _prepare_cameras(
    cameras: Sequence[Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack camera intrinsics/extrinsics into tensors on ``device``."""
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    return K, R, t


def evaluate_mpjpe_at_k(
    model: torch.nn.Module,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras: Sequence[Any],
    *,
    k_values: Sequence[int],
    clip_len: int,
    num_subsets_per_k: Optional[int] = None,
    seed: int = 42,
    align: str = "none",
    device: Optional[torch.device] = None,
    hardened: bool = False,
    n_views_max: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate ``model`` for variable view counts using the MPJPE@k protocol.

    Args:
        model: A fixed-view fusion model that accepts ``x, K, R, t``.
        points_2d: (T, V, J, 2) 2D keypoints.
        confidences: (T, V, J) confidence scores.
        joints_3d: (T, J, 3) ground-truth 3D poses.
        cameras: Sequence of ``V`` camera objects with ``K``, ``R``, ``t`` attrs.
        k_values: View counts to evaluate.
        clip_len: Temporal clip length.
        num_subsets_per_k: Number of subsets per ``k``; ``None`` enumerates all.
        seed: RNG seed for subset sampling.
        align: ``"none"``, ``"pa"``, or ``"root"``.
        device: PyTorch device; inferred if not provided.
        hardened: Use :class:`HardenedVariableViewInferenceWrapper`.
        n_views_max: Fixed view count the model expects.  If None, use the
            dataset's actual view count ``V``.

    Returns:
        Mapping ``k -> dict(metrics)``.  Metrics include ``mpjpe``,
        ``pa_mpjpe``, ``root_rel_mpjpe``, ``temporal_jerk``, ``n_subsets``,
        ``subsets`` (list of tuples), and ``per_subset_mpjpe``.
    """
    if device is None:
        device = next(model.parameters()).device
    T, V, J = points_2d.shape[:3]
    k_values = sorted(set(k_values))

    K, R, t = _prepare_cameras(cameras, device)
    wrapper_cls = HardenedVariableViewInferenceWrapper if hardened else VariableViewInferenceWrapper
    wrapper = wrapper_cls(model)
    rng = np.random.default_rng(seed)

    all_subsets = generate_view_subsets(V, k_values, num_subsets_per_k=num_subsets_per_k, seed=seed)
    results: Dict[int, Dict[str, Any]] = {}

    for k, subsets in all_subsets.items():
        per_subset_errors: List[float] = []
        subset_results: List[Dict[str, float]] = []

        for subset in subsets:
            active = torch.zeros(V, dtype=torch.bool)
            active[list(subset)] = True

            clip_preds: List[np.ndarray] = []
            clip_gts: List[np.ndarray] = []
            starts = list(range(0, T - clip_len + 1, clip_len))
            if not starts:
                starts = [0]
                # If the sequence is shorter than clip_len we still evaluate once,
                # padding via prepare_variable_view_input is handled below.

            for start in starts:
                end = min(start + clip_len, T)
                x_clip = torch.from_numpy(np.concatenate([
                    points_2d[start:end],
                    confidences[start:end, ..., None],
                ], axis=-1)).float().to(device)
                x_padded, Kp, Rp, tp, _ = prepare_variable_view_input(
                    x_clip, K, R, t, active, n_views_max=(n_views_max if n_views_max is not None else V)
                )
                with torch.no_grad():
                    pred = wrapper.model(x_padded.unsqueeze(0), K=Kp, R=Rp, t=tp)[0]
                pred = pred.squeeze(0).cpu().numpy()  # (T_clip, J, 3)
                clip_preds.append(pred)
                clip_gts.append(joints_3d[start:end])

            pred_all = np.concatenate(clip_preds, axis=0)
            gt_all = np.concatenate(clip_gts, axis=0)

            metrics = compute_mpjpe_at_k(pred_all, gt_all, k=k, align=align)
            per_subset_errors.append(metrics["mpjpe"])
            subset_results.append(metrics)

        if per_subset_errors:
            results[k] = {
                "k": k,
                "mpjpe": float(np.mean(per_subset_errors)),
                "std_mm": float(np.std(per_subset_errors)),
                "n_subsets": len(per_subset_errors),
                "subsets": [list(s) for s in subsets],
                "per_subset": subset_results,
            }
            # Aggregate other metrics across subsets.
            for key in ("pa_mpjpe", "root_rel_mpjpe", "temporal_jerk"):
                values = [r[key] for r in subset_results]
                results[k][key] = float(np.mean(values))
    return results


def print_mpjpe_at_k_table(
    results: Dict[int, Dict[str, Any]],
    label: str = "MPJPE@k",
) -> None:
    """Print a compact table of MPJPE@k results."""
    print(f"{label} results (mm):")
    header = f"{'k':>3} {'MPJPE':>8} {'PA-MPJPE':>10} {'Root-Rel':>10} {'Jerk':>8} {'Subsets':>8}"
    print(header)
    print("-" * len(header))
    for k in sorted(results):
        r = results[k]
        print(
            f"{k:>3} "
            f"{r['mpjpe']:>8.2f} "
            f"{r.get('pa_mpjpe', float('nan')):>10.2f} "
            f"{r.get('root_rel_mpjpe', float('nan')):>10.2f} "
            f"{r.get('temporal_jerk', 0.0):>8.2f} "
            f"{r['n_subsets']:>8d}"
        )


def write_mpjpe_at_k_csv(
    results: Dict[int, Dict[str, Any]],
    path: Path | str,
) -> None:
    """Write MPJPE@k results to a CSV file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("k,mpjpe,pa_mpjpe,root_rel_mpjpe,temporal_jerk,n_subsets\n")
        for k in sorted(results):
            r = results[k]
            f.write(
                f"{k},{r['mpjpe']:.4f},"
                f"{r.get('pa_mpjpe', float('nan')):.4f},"
                f"{r.get('root_rel_mpjpe', float('nan')):.4f},"
                f"{r.get('temporal_jerk', 0.0):.4f},"
                f"{r['n_subsets']}\n"
            )
    print(f"Saved MPJPE@k CSV to {out_path}")


def write_mpjpe_at_k_json(
    results: Dict[int, Dict[str, Any]],
    path: Path | str,
) -> None:
    """Write MPJPE@k results to a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved MPJPE@k JSON to {out_path}")
