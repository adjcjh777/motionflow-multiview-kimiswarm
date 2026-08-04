"""Standard 3D human pose evaluation metrics.

Summary (2025-08-04 swarm task):
    Added batched/per-joint/per-view breakdown variants for MPJPE, PA-MPJPE,
    PCK, and AUC. The scalar helpers (mpjpe, pa_mpjpe, pck) remain unchanged
    so existing callers keep working.
"""

from typing import Dict, Tuple, Union
import numpy as np


ScalarOrArray = Union[float, np.ndarray]


def _to_array(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    return pred, gt


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Per Joint Position Error (mm). Assumes inputs in mm."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def mpjpe_batch(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean MPJPE over a batch.

    Args:
        pred: (..., J, 3) predicted joints.
        gt: (..., J, 3) ground-truth joints.

    Returns:
        Scalar mean error.
    """
    pred, gt = _to_array(pred, gt)
    dists = np.linalg.norm(pred - gt, axis=-1)  # (..., J)
    return float(dists.mean())


def per_joint_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint MPJPE.

    Args:
        pred: (..., J, 3)
        gt: (..., J, 3)

    Returns:
        (J,) array of per-joint mean errors.
    """
    pred, gt = _to_array(pred, gt)
    dists = np.linalg.norm(pred - gt, axis=-1)  # (..., J)
    return dists.reshape(-1, dists.shape[-1]).mean(axis=0)  # (J,)


def per_view_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-view MPJPE.

    Args:
        pred: (V, ..., 3) or (..., V, ..., 3). Treats first axis as views.
        gt: same shape.

    Returns:
        (V,) array of per-view mean errors.
    """
    pred, gt = _to_array(pred, gt)
    dists = np.linalg.norm(pred - gt, axis=-1)  # (V, ...)
    return dists.reshape(dists.shape[0], -1).mean(axis=1)  # (V,)


def _align_rigid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Align X to Y using rigid Procrustes (translation + rotation, no scale)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return (Xc @ R) + Y.mean(axis=0)


def _align_rigid_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-align each frame of pred to the corresponding gt frame.

    Args:
        pred: (B, J, 3)
        gt: (B, J, 3)

    Returns:
        (B, J, 3) aligned predictions.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    aligned = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        aligned[i] = _align_rigid(pred[i], gt[i])
    return aligned


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """PA-MPJPE: Procrustes-aligned MPJPE.

    Supports:
        * single skeleton: pred/gt are (J, 3)
        * batched skeletons: pred/gt are (B, J, 3) or any shape that can be
          reshaped to (B, J, 3)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 2:
        pred_aligned = _align_rigid(pred, gt)
        return mpjpe(pred_aligned, gt)
    # batched
    pred = pred.reshape(-1, gt.shape[-2], 3)
    gt = gt.reshape(-1, gt.shape[-2], 3)
    pred_aligned = _align_rigid_batch(pred, gt)
    return float(np.mean(np.linalg.norm(pred_aligned - gt, axis=-1)))


def pa_mpjpe_per_joint(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint PA-MPJPE.

    Args:
        pred: (B, J, 3) or (J, 3).
        gt: same shape.

    Returns:
        (J,) array of per-joint PA-MPJPE.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 2:
        pred_aligned = _align_rigid(pred, gt)
        return per_joint_mpjpe(pred_aligned[None, ...], gt[None, ...])
    pred_aligned = _align_rigid_batch(pred, gt)
    return per_joint_mpjpe(pred_aligned, gt)


def pck(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Percentage of Correct Keypoints at given threshold (same unit as input)."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    dists = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean(dists < threshold))


def pck_batch(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """PCK over a batch of predictions.

    Args:
        pred: (..., J, 3)
        gt: (..., J, 3)
        threshold: distance threshold in the same unit as the data.

    Returns:
        Scalar PCK.
    """
    pred, gt = _to_array(pred, gt)
    dists = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean(dists < threshold))


def pck_per_joint(pred: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    """Per-joint PCK.

    Args:
        pred: (..., J, 3)
        gt: (..., J, 3)
        threshold: distance threshold.

    Returns:
        (J,) array of per-joint PCKs.
    """
    pred, gt = _to_array(pred, gt)
    dists = np.linalg.norm(pred - gt, axis=-1)  # (..., J)
    correct = dists < threshold
    return correct.reshape(-1, correct.shape[-1]).mean(axis=0)


def pck_auc(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: np.ndarray = None,
    max_threshold: float = 150.0,
    n_points: int = 100,
    per_joint: bool = False,
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Area Under the Curve of PCK over a range of thresholds.

    The AUC is computed with the trapezoidal rule and normalized by the range
    of thresholds so that a perfect predictor scores 1.0.

    Args:
        pred: (..., J, 3) predictions.
        gt: (..., J, 3) ground truth.
        thresholds: explicit thresholds in the same unit as the data. If None,
            uses np.linspace(0, max_threshold, n_points).
        max_threshold: upper bound when thresholds is not provided.
        n_points: number of thresholds to sample.
        per_joint: if True, return per-joint AUC arrays instead of a scalar.

    Returns:
        auc: scalar AUC (or (J,) if per_joint=True).
        thresholds: the threshold array used.
        pcks: PCK values at each threshold, shape (N,) or (J, N).
    """
    pred, gt = _to_array(pred, gt)
    if thresholds is None:
        thresholds = np.linspace(0.0, max_threshold, n_points)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)
    dists = np.linalg.norm(pred - gt, axis=-1)  # (..., J)

    if per_joint:
        dists_flat = dists.reshape(-1, dists.shape[-1])  # (B, J)
        pcks = np.empty((dists_flat.shape[1], len(thresholds)), dtype=np.float64)
        for t_idx, thr in enumerate(thresholds):
            pcks[:, t_idx] = np.mean(dists_flat < thr, axis=0)
        aucs = np.trapz(pcks, thresholds, axis=1) / (thresholds[-1] - thresholds[0])
        return aucs, thresholds, pcks

    pcks = np.asarray([np.mean(dists < thr) for thr in thresholds])
    auc = np.trapz(pcks, thresholds) / (thresholds[-1] - thresholds[0])
    return auc, thresholds, pcks


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    pck_thresholds: np.ndarray = None,
    pck_auc_max: float = 150.0,
    pck_auc_points: int = 100,
) -> Dict[str, ScalarOrArray]:
    """Compute a full metrics report for a batch of 3D predictions.

    Args:
        pred: (B, J, 3) predicted 3D poses.
        gt: (B, J, 3) ground-truth 3D poses.
        pck_thresholds: dict of {name: threshold} or a flat array of thresholds.
            Defaults to {50mm, 100mm, 150mm}.
        pck_auc_max: max threshold (mm) for AUC computation.
        pck_auc_points: number of thresholds for AUC.

    Returns:
        dict with keys: mpjpe, pa_mpjpe, pck@{name}, pck_auc, and per-joint
        variants.
    """
    pred, gt = _to_array(pred, gt)
    if pred.ndim != 3:
        pred = pred.reshape(-1, pred.shape[-2], 3)
        gt = gt.reshape(-1, gt.shape[-2], 3)

    if pck_thresholds is None:
        pck_thresholds = {"50mm": 50.0, "100mm": 100.0, "150mm": 150.0}
    elif isinstance(pck_thresholds, np.ndarray):
        pck_thresholds = {f"{t:.1f}mm": float(t) for t in pck_thresholds}

    report: Dict[str, ScalarOrArray] = {
        "mpjpe": mpjpe_batch(pred, gt),
        "pa_mpjpe": pa_mpjpe(pred, gt),
    }

    for name, thr in pck_thresholds.items():
        report[f"pck@{name}"] = pck_batch(pred, gt, thr)
        report[f"pck@{name}_per_joint"] = pck_per_joint(pred, gt, thr)

    auc, _, _ = pck_auc(pred, gt, max_threshold=pck_auc_max, n_points=pck_auc_points)
    report["pck_auc"] = auc
    report["pck_auc_per_joint"] = pck_auc(
        pred, gt, max_threshold=pck_auc_max, n_points=pck_auc_points, per_joint=True
    )[0]
    report["per_joint_mpjpe"] = per_joint_mpjpe(pred, gt)
    report["per_joint_pa_mpjpe"] = pa_mpjpe_per_joint(pred, gt)

    return report


def summarize_metrics(report: Dict[str, ScalarOrArray]) -> str:
    """Return a compact printable summary of a metrics report."""
    lines = [
        f"MPJPE: {report['mpjpe']:.4f}",
        f"PA-MPJPE: {report['pa_mpjpe']:.4f}",
    ]
    for key in sorted(report):
        if key.startswith("pck@") and "_per_joint" not in key:
            lines.append(f"{key.upper()}: {report[key]:.4f}")
    if "pck_auc" in report:
        lines.append(f"PCK AUC (0-150mm): {report['pck_auc']:.4f}")
    return "\n".join(lines)
