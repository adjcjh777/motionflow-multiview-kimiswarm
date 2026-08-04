"""Standard 3D human pose evaluation metrics."""

import numpy as np


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Per Joint Position Error (mm). Assumes inputs in mm."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


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


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """PA-MPJPE: Procrustes-aligned MPJPE."""
    pred_aligned = _align_rigid(pred, gt)
    return mpjpe(pred_aligned, gt)


def pck(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Percentage of Correct Keypoints at given threshold (same unit as input)."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    dists = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean(dists < threshold))
