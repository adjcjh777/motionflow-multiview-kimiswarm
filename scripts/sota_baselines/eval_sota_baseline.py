#!/usr/bin/env python
"""Evaluate a SOTA baseline's 3D predictions against the H36M true-GT labels.

The predictions file must contain 3D joints for the validation/test subjects.
Supported formats:

- .npz with keys ``joints_3d`` (F, J, 3) and optionally ``frame_indices``
- .pkl / .pickle dict with keys ``joints_3d`` (F, J, 3) and optionally
  ``frame_indices``

The ground-truth can be either a single H36M true-GT ``*_multiview_m.npz``
file or the manifest produced by ``convert_to_mvpose_format.py``.

Examples
--------
Evaluate VoxelPose-style predictions on S9::

    python scripts/sota_baselines/eval_sota_baseline.py \
        --pred tmp/sota_baselines/voxelpose_data/h36m_true_gt_pred_s9.npz \
        --gt data/h36m_true_gt/s_09_acts_02_03_..._multiview_m.npz \
        --out_json outputs/sota_baselines/voxelpose_h36m_s9_metrics.json

"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Allow importing project modules when run from repo root.
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics


def load_predictions(path: Path) -> Dict[str, Any]:
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    if path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported prediction file format: {path}")


def load_ground_truth(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "joints_3d": data["joints_3d"],
        "points_2d": data["points_2d"],
        "confidences": data["confidences"],
        "camera_K": data["camera_K"],
        "camera_R": data["camera_R"],
        "camera_t": data["camera_t"],
    }


def align_predictions_to_gt(
    pred: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    """Procrustes alignment for PA-MPJPE computation.

    pred and gt are (F, J, 3).
    Returns pred aligned to gt per-frame.
    """
    aligned = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        p = pred[i]
        g = gt[i]
        # Center.
        p_c = p - p.mean(axis=0, keepdims=True)
        g_c = g - g.mean(axis=0, keepdims=True)
        # Optimal scale.
        s = np.linalg.norm(g_c) / (np.linalg.norm(p_c) + 1e-8)
        # Rotation via SVD.
        H = p_c.T @ g_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        aligned[i] = s * (p_c @ R) + g.mean(axis=0, keepdims=True)
    return aligned


def evaluate_predictions(
    pred: np.ndarray,
    gt: np.ndarray,
    output_json: Path | None = None,
    method_name: str = "sota_baseline",
) -> Dict[str, float]:
    """Compute MPJPE, PA-MPJPE, and related metrics."""
    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} does not match GT shape {gt.shape}"
        )

    # Use project metrics where possible.
    report = compute_all_metrics(pred, gt)

    # Procrustes alignment for PA-MPJPE.
    aligned = align_predictions_to_gt(pred, gt)
    pa_errors = np.linalg.norm(aligned - gt, axis=-1)  # (F, J)
    pa_mpjpe = float(pa_errors.mean())

    result = {
        "method": method_name,
        "num_frames": pred.shape[0],
        "num_joints": pred.shape[1],
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": pa_mpjpe,
        "mpjpe_per_joint_mm": [
            float(report["mpjpe_per_joint"][j]) for j in range(pred.shape[1])
        ],
    }

    if "pck@50mm" in report:
        result["pck_50mm"] = float(report["pck@50mm"])
    if "pck_auc" in report:
        result["pck_auc"] = float(report["pck_auc"])

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {output_json}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Path to predictions .npz or .pkl")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground-truth .npz")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--method", type=str, default="sota_baseline", help="Method name for metrics.")
    args = parser.parse_args()

    pred_data = load_predictions(args.pred)
    gt_data = load_ground_truth(args.gt)

    pred_joints = pred_data["joints_3d"]
    gt_joints = gt_data["joints_3d"]

    # If frame indices are provided, slice the GT to match.
    if "frame_indices" in pred_data:
        indices = pred_data["frame_indices"]
        gt_joints = gt_joints[indices]

    # Convert to mm if the predictions are in metres.
    if np.abs(gt_joints).max() > 10 and np.abs(pred_joints).max() < 10:
        pred_joints = pred_joints * 1000.0
    elif np.abs(gt_joints).max() < 10 and np.abs(pred_joints).max() > 10:
        pred_joints = pred_joints / 1000.0

    result = evaluate_predictions(
        pred_joints, gt_joints, output_json=args.out_json, method_name=args.method
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
