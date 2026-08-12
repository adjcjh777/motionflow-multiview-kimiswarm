#!/usr/bin/env python
"""Evaluate MVPose H36M adapter predictions against the converted val pickle.

The adapter and the converted pickle both use COCO17 joint order. Because the
five COCO17 facial keypoints are approximated from the single H36M Head joint,
we report metrics on both the full 17-joint set and a body-only 12-joint subset
(joint indices 5-16) that avoids degenerate duplicates.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _align_frame(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Procrustes-align a single-frame pose p to g."""
    p_c = p - p.mean(axis=0, keepdims=True)
    g_c = g - g.mean(axis=0, keepdims=True)
    s = np.linalg.norm(g_c) / (np.linalg.norm(p_c) + 1e-8)
    H = p_c.T @ g_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return s * (p_c @ R) + g.mean(axis=0, keepdims=True)


def _per_frame_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def _per_frame_pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    aligned = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        aligned[i] = _align_frame(pred[i], gt[i])
    return float(np.linalg.norm(aligned - gt, axis=-1).mean())


def _evaluate_subset(pred: np.ndarray, gt: np.ndarray, indices: np.ndarray | None) -> Dict[str, float]:
    if indices is not None:
        pred = pred[:, indices]
        gt = gt[:, indices]
    return {
        "mpjpe_mm": _per_frame_mpjpe(pred, gt) * 1000.0,
        "pa_mpjpe_mm": _per_frame_pa_mpjpe(pred, gt) * 1000.0,
        "num_frames": pred.shape[0],
        "num_joints": pred.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pkl", type=Path, required=True,
                        help="Converted MVPose val pickle.")
    parser.add_argument("--pred_dir", type=Path, required=True,
                        help="Directory containing per-sequence prediction .npz files.")
    parser.add_argument("--out_json", type=Path, default=None,
                        help="Optional JSON output path.")
    args = parser.parse_args()

    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)

    sequences: List[dict] = data.get("sequences", data.get("val", []))
    per_seq: Dict[str, Dict[str, Any]] = {}
    total_frames = 0
    weighted_all = {"mpjpe": 0.0, "pa_mpjpe": 0.0, "frames": 0}
    weighted_body = {"mpjpe": 0.0, "pa_mpjpe": 0.0, "frames": 0}
    body_indices = np.arange(5, 17)  # body-only COCO indices

    for idx, seq in enumerate(sequences):
        subject = seq.get("subject", idx)
        actions = seq.get("actions", [])
        if actions:
            seq_name = f"s_{subject:02d}_acts_{'_'.join(str(a) for a in actions)}"
        else:
            seq_name = f"s_{subject:02d}_seq_{idx}"

        pred_path = args.pred_dir / f"{seq_name}_pred.npz"
        pred = np.load(pred_path)["joints_3d"]
        gt = seq["joints_3d"]

        if np.abs(gt).max() > 10 and np.abs(pred).max() < 10:
            pred = pred * 1000.0
        elif np.abs(gt).max() < 10 and np.abs(pred).max() > 10:
            pred = pred / 1000.0

        all_metrics = _evaluate_subset(pred, gt, None)
        body_metrics = _evaluate_subset(pred, gt, body_indices)

        per_seq[seq_name] = {
            "all_17_joints": all_metrics,
            "body_12_joints": body_metrics,
        }

        n = all_metrics["num_frames"]
        total_frames += n
        weighted_all["mpjpe"] += all_metrics["mpjpe_mm"] * n
        weighted_all["pa_mpjpe"] += all_metrics["pa_mpjpe_mm"] * n
        weighted_body["mpjpe"] += body_metrics["mpjpe_mm"] * n
        weighted_body["pa_mpjpe"] += body_metrics["pa_mpjpe_mm"] * n
        weighted_all["frames"] += n
        weighted_body["frames"] += n

        print(
            f"{seq_name}: all17 MPJPE={all_metrics['mpjpe_mm']:.2f} mm "
            f"PA={all_metrics['pa_mpjpe_mm']:.2f} mm; "
            f"body12 MPJPE={body_metrics['mpjpe_mm']:.2f} mm "
            f"PA={body_metrics['pa_mpjpe_mm']:.2f} mm"
        )

    combined = {
        "all_17_joints": {
            "mpjpe_mm": weighted_all["mpjpe"] / weighted_all["frames"],
            "pa_mpjpe_mm": weighted_all["pa_mpjpe"] / weighted_all["frames"],
        },
        "body_12_joints": {
            "mpjpe_mm": weighted_body["mpjpe"] / weighted_body["frames"],
            "pa_mpjpe_mm": weighted_body["pa_mpjpe"] / weighted_body["frames"],
        },
        "num_frames": total_frames,
    }

    report = {
        "input_pkl": str(args.input_pkl),
        "pred_dir": str(args.pred_dir),
        "per_sequence": per_seq,
        "combined": combined,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {args.out_json}")

    print(
        f"Combined (frame-weighted, {total_frames} frames):\n"
        f"  all17  MPJPE={combined['all_17_joints']['mpjpe_mm']:.2f} mm, "
        f"PA-MPJPE={combined['all_17_joints']['pa_mpjpe_mm']:.2f} mm\n"
        f"  body12 MPJPE={combined['body_12_joints']['mpjpe_mm']:.2f} mm, "
        f"PA-MPJPE={combined['body_12_joints']['pa_mpjpe_mm']:.2f} mm"
    )


if __name__ == "__main__":
    main()
