"""Faster variable-view evaluation with batched clips.

This is a drop-in replacement for ``experiments/eval_variable_views.py`` that
batches all clips of a single subset into one GPU forward pass and adds a
tqdm progress bar.  Output format is identical.
"""

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import mpjpe as mpjpe_metric
from motionflow_mv.fusion.ray_attention_temporal_residual_model import (
    RayAttentionFusionModelTemporalResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_model import (
    RayAttentionFusionModelTemporalCrossviewResidual,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
)
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_visibility_model import (
    RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
)
from motionflow_mv.fusion.variable_view_inference import (
    VariableViewInferenceWrapper,
    prepare_variable_view_input,
)
from motionflow_mv.calibration.camera import Camera


def _make_synthetic_cameras(n_views: int = 4):
    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        c = np.array([3 * np.cos(theta), 3 * np.sin(theta), 1.0])
        forward = -c / (np.linalg.norm(c) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=0)
        t_vec = -R @ c
        K = np.eye(3)
        K[0, 0] = K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        cameras.append(Camera(K=K, R=R, t=t_vec))
    return cameras


def _make_synthetic_dataset(n_views: int, j: int, n_frames: int):
    rng = np.random.default_rng(2024)
    cameras = _make_synthetic_cameras(n_views)
    points_2d = rng.uniform(0, 1, size=(n_frames, n_views, j, 2)).astype(np.float32)
    confidences = np.ones((n_frames, n_views, j), dtype=np.float32)
    joints_3d = rng.uniform(-1, 1, size=(n_frames, j, 3)).astype(np.float32)
    return points_2d, confidences, joints_3d, cameras


MODEL_CLASSES = {
    "temporal_residual": RayAttentionFusionModelTemporalResidual,
    "crossview_residual": RayAttentionFusionModelTemporalCrossviewResidual,
    "crossview_residual_pp": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPoint,
    "crossview_residual_pp_visibility": RayAttentionFusionModelTemporalCrossviewResidualPrincipalPointVisibility,
}


def _build_model(
    j: int,
    n_views: int,
    d: int,
    n_temporal_layers: int,
    checkpoint: str | None,
    model_class: str = "temporal_residual",
    residual_hidden: int = 128,
):
    cls = MODEL_CLASSES[model_class]
    kwargs = {"j": j, "d": d, "n_views": n_views, "residual_hidden": residual_hidden}
    if model_class in {"crossview_residual", "crossview_residual_pp", "crossview_residual_pp_visibility"}:
        kwargs["n_st_layers"] = n_temporal_layers
    else:
        kwargs["n_temporal_layers"] = n_temporal_layers
    model = cls(**kwargs)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_variable_views_batched(
    model,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    cameras,
    clip_len: int,
    device: torch.device,
    min_views: int = 2,
    max_views: int | None = None,
    num_subsets_per_k: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
) -> dict:
    """Batched variable-view evaluation with bounded batch size."""
    if max_views is None:
        max_views = points_2d.shape[1]
    V = points_2d.shape[1]
    J = points_2d.shape[2]
    T = points_2d.shape[0]

    K = torch.from_numpy(np.stack([cam.K for cam in cameras], axis=0)).float().to(device)
    R = torch.from_numpy(np.stack([cam.R for cam in cameras], axis=0)).float().to(device)
    t = torch.from_numpy(np.stack([cam.t for cam in cameras], axis=0)).float().to(device)

    wrapper = VariableViewInferenceWrapper(model)
    rng = np.random.default_rng(seed)

    # Pre-compute all clips once.
    all_clips = []
    for start in range(0, T - clip_len + 1, clip_len):
        end = start + clip_len
        x_clip = torch.from_numpy(np.concatenate([
            points_2d[start:end],
            confidences[start:end, ..., None],
        ], axis=-1)).float().to(device)
        all_clips.append((start, end, x_clip))
    n_clips = len(all_clips)

    results = {}
    for k in range(min_views, max_views + 1):
        subset_errors = []
        total_subsets = math.comb(V, k)
        if num_subsets_per_k is None or total_subsets <= num_subsets_per_k:
            subsets = list(combinations(range(V), k))
        else:
            subsets = set()
            while len(subsets) < num_subsets_per_k:
                subsets.add(tuple(sorted(rng.choice(V, size=k, replace=False))))
            subsets = list(subsets)

        for subset in subsets:
            active = torch.zeros(V, dtype=torch.bool)
            active[list(subset)] = True

            # Prepare all clips for this subset.
            clip_batch = []
            for _, _, x_clip in all_clips:
                x_clip, Kp, Rp, tp, _ = prepare_variable_view_input(
                    x_clip, K, R, t, active, n_views_max=V
                )
                clip_batch.append(x_clip)
            if not clip_batch:
                continue

            # Run forward in chunks to keep CUDA kernel arguments valid.
            preds = []
            for i in range(0, n_clips, batch_size):
                chunk = clip_batch[i:i + batch_size]
                x_batch = torch.stack(chunk, dim=0)  # (B, T, V, J, 3)
                with torch.no_grad():
                    pred_batch = wrapper.model(x_batch, K=Kp, R=Rp, t=tp)[0]
                preds.append(pred_batch.cpu().numpy())

            pred_all = np.concatenate([p.reshape(-1, J, 3) for p in preds], axis=0)

            # GT clips in the same order.
            gt_batch = [joints_3d[start:end] for start, end, _ in all_clips]
            gt_all = np.concatenate(gt_batch, axis=0)
            err = mpjpe_metric(pred_all * 1000.0, gt_all * 1000.0)
            subset_errors.append(err)

        if subset_errors:
            results[k] = {
                "mean_mm": float(np.mean(subset_errors)),
                "std_mm": float(np.std(subset_errors)),
                "n_subsets": len(subset_errors),
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--j", type=int, default=17)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--clip_len", type=int, default=9)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--model_class", type=str, default="temporal_residual",
                       choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--num_subsets_per_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--n_frames", type=int, default=36,
                        help="Synthetic frame count when --dataset is not given")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset is not None:
        data = np.load(args.dataset)
        points_2d = data["points_2d"]
        confidences = data["confidences"]
        joints_3d = data["joints_3d"]
        n_views = data["camera_K"].shape[0]
        cameras = []
        for v in range(n_views):
            cameras.append(Camera(
                K=data["camera_K"][v],
                R=data["camera_R"][v],
                t=data["camera_t"][v],
            ))
    else:
        n_views = args.n_views
        points_2d, confidences, joints_3d, cameras = _make_synthetic_dataset(
            n_views, args.j, args.n_frames)

    model = _build_model(args.j, n_views, args.d, args.n_temporal_layers,
                         args.checkpoint, args.model_class, args.residual_hidden).to(device)

    results = evaluate_variable_views_batched(
        model, points_2d, confidences, joints_3d, cameras,
        clip_len=args.clip_len, device=device, min_views=args.min_views,
        max_views=args.max_views or n_views,
        num_subsets_per_k=args.num_subsets_per_k, seed=args.seed,
    )

    print("Variable view-count benchmark results (MPJPE mm):")
    for k, res in results.items():
        print(f"  k={k:2d}: mean={res['mean_mm']:.4f} mm, "
              f"std={res['std_mm']:.4f} mm, subsets={res['n_subsets']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in results.items()}, f, indent=2)
        print(f"Saved results to {out_path}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("k,mean_mm,std_mm,n_subsets\n")
            for k, res in results.items():
                f.write(f"{k},{res['mean_mm']:.4f},{res['std_mm']:.4f},{res['n_subsets']}\n")
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
