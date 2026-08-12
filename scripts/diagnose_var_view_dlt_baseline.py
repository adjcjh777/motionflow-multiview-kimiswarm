"""Diagnose variable-view k<4 failure by comparing model output to direct DLT.

This script loads a v25 OmniMultiViewFusionV5 checkpoint and evaluates:
  1. final model output (via HardenedVariableViewInferenceWrapper)
  2. direct confidence-weighted DLT using only the active views

on a small H36M true-GT subset for k=2,3,4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq
from motionflow_mv.fusion.variable_view_inference import (
    HardenedVariableViewInferenceWrapper,
)
from experiments.eval_variable_views import _build_omniview_v5_model, _load_config, _load_npz_dataset


def evaluate_direct_dlt(
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras,
    clip_len: int,
    device: torch.device,
    k_values=(2, 3, 4),
):
    """Direct confidence-weighted DLT baseline for variable views."""
    T, V, J, _ = points_2d.shape
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)
    Rt = torch.cat([R, t[..., None]], dim=-1)  # (V, 3, 4)
    P = K @ Rt  # (V, 3, 4)

    results = {}
    for k in k_values:
        subset = list(range(k))
        active = torch.zeros(V, dtype=torch.bool)
        active[subset] = True
        active_indices = torch.where(active)[0]

        clip_preds = []
        starts = list(range(0, T - clip_len + 1, clip_len))
        if not starts:
            starts = [0]
        for start in starts:
            end = min(start + clip_len, T)
            x_clip = torch.from_numpy(np.concatenate([
                points_2d[start:end],
                confidences[start:end, ..., None],
            ], axis=-1)).float().to(device)
            x_active = x_clip[:, active_indices, :, :]
            P_active = P[active_indices]
            points_2d_active = x_active[..., :2]
            confidences_active = x_active[..., 2]
            pred = triangulate_dlt_batched_lstsq(points_2d_active, P_active, confidences_active)
            clip_preds.append(pred.cpu().numpy())
        pred_all = np.concatenate(clip_preds, axis=0)
        mpjpe_val = float(mpjpe_metric(pred_all * 1000.0, joints_3d[: len(pred_all)] * 1000.0))
        results[k] = mpjpe_val
    return results


def evaluate_model_output(
    model,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras,
    clip_len: int,
    device: torch.device,
    k_values=(2, 3, 4),
):
    """Model output via HardenedVariableViewInferenceWrapper."""
    T, V, J, _ = points_2d.shape
    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)

    wrapper = HardenedVariableViewInferenceWrapper(model)
    results = {}
    for k in k_values:
        clip_preds = []
        starts = list(range(0, T - clip_len + 1, clip_len))
        if not starts:
            starts = [0]
        for start in starts:
            end = min(start + clip_len, T)
            x_clip = torch.from_numpy(np.concatenate([
                points_2d[start:end],
                confidences[start:end, ..., None],
            ], axis=-1)).float().to(device)
            x_clip = x_clip.unsqueeze(0)  # (1, T_clip, V, J, 3)
            with torch.no_grad():
                pred = wrapper(x_clip, K=K, R=R, t=t, active_views=k)[0]
            clip_preds.append(pred.squeeze(0).cpu().numpy())
        pred_all = np.concatenate(clip_preds, axis=0)
        mpjpe_val = float(mpjpe_metric(pred_all * 1000.0, joints_3d[: len(pred_all)] * 1000.0))
        results[k] = mpjpe_val
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--output_json", default="/tmp/var_view_dlt_baseline.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _load_config(args.config)
    n_joints = np.load(args.dataset)["joints_3d"].shape[1]
    # The checkpoint/config defines the model's fixed view count; the variable-view
    # wrapper pads the actual dataset views up to that count.
    n_views = config.get("n_views", np.load(args.dataset)["camera_K"].shape[0])
    model = _build_omniview_v5_model(config, args.checkpoint, n_joints, n_views, device)

    points_2d, confidences, joints_3d, cameras = _load_npz_dataset(args.dataset)

    model_results = evaluate_model_output(model, points_2d, confidences, joints_3d, cameras, args.clip_len, device)
    dlt_results = evaluate_direct_dlt(points_2d, confidences, joints_3d, cameras, args.clip_len, device)

    print("MPJPE@k (mm):")
    print(f"{'k':>3} {'model':>12} {'direct_dlt':>12}")
    for k in sorted(model_results):
        print(f"{k:>3} {model_results[k]:>12.4f} {dlt_results[k]:>12.4f}")

    out = {"model": model_results, "direct_dlt": dlt_results}
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
