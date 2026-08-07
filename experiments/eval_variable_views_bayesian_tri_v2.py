"""Variable-view MPJPE@k for the Bayesian Tri v2 model.

Example
-------
    python experiments/eval_variable_views_bayesian_tri_v2.py \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/bayesian_tri_v2_stabilized_mpiinf3dhp.pth \
        --output_json outputs/variable_views_bayesian_tri_v2.json
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
from motionflow_mv.fusion.ray_attention_temporal_crossview_residual_principal_point_bayesian_tri_model import (
    RayAttentionFusionModelBayesianTriV2,
)
from motionflow_mv.fusion.variable_view_inference import (
    VariableViewInferenceWrapper,
    prepare_variable_view_input,
)


def evaluate_variable_views(
    model,
    points_2d: np.ndarray,
    confidences: np.ndarray,
    joints_3d: np.ndarray,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    clip_len: int,
    device: torch.device,
    min_views: int = 2,
    max_views: int | None = None,
    num_subsets_per_k: int = 10,
    seed: int = 42,
    batch_size: int = 8,
):
    V = points_2d.shape[1]
    J = points_2d.shape[2]
    T = points_2d.shape[0]

    wrapper = VariableViewInferenceWrapper(model)
    rng = np.random.default_rng(seed)

    all_clips = []
    for start in range(0, T - clip_len + 1, clip_len):
        end = start + clip_len
        x_clip = torch.from_numpy(
            np.concatenate([points_2d[start:end], confidences[start:end, ..., None]], axis=-1)
        ).float()
        all_clips.append((start, end, x_clip))

    results = {}
    for k in range(min_views, (max_views or V) + 1):
        subsets = list(combinations(range(V), k))
        if len(subsets) > num_subsets_per_k:
            idx = rng.choice(len(subsets), size=num_subsets_per_k, replace=False)
            subsets = [subsets[i] for i in idx]

        errors = []
        for subset in subsets:
            active = torch.zeros(V, dtype=torch.bool)
            active[list(subset)] = True

            preds = []
            gt_list = []
            for start, end, x_clip in all_clips:
                x_clip, Kp, Rp, tp, _ = prepare_variable_view_input(
                    x_clip, K, R, t, active, n_views_max=V
                )
                x_clip = x_clip.to(device)
                Kp = Kp.to(device)
                Rp = Rp.to(device)
                tp = tp.to(device)
                with torch.no_grad():
                    pred = wrapper.model(x_clip.unsqueeze(0), K=Kp, R=Rp, t=tp)[0]
                preds.append(pred[0].cpu().numpy())
                gt_list.append(joints_3d[start:end])

            pred_all = np.concatenate(preds, axis=0)
            gt_all = np.concatenate(gt_list, axis=0)
            err = mpjpe_metric(pred_all * 1000.0, gt_all * 1000.0)
            errors.append(float(err))

        results[k] = {
            "mean_mm": float(np.mean(errors)) if errors else None,
            "std_mm": float(np.std(errors)) if errors else None,
            "n_subsets": len(errors),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="outputs/variable_views_bayesian_tri_v2.json")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--residual_hidden", type=int, default=256)
    parser.add_argument("--n_st_layers", type=int, default=3)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--num_subsets_per_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])

    model = RayAttentionFusionModelBayesianTriV2(
        j=j,
        d=args.d,
        n_views=n_views,
        n_st_layers=args.n_st_layers,
        residual_hidden=args.residual_hidden,
        return_pp_delta=True,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    K = torch.from_numpy(data["camera_K"]).float()
    R = torch.from_numpy(data["camera_R"]).float()
    t = torch.from_numpy(data["camera_t"]).float()

    results = evaluate_variable_views(
        model,
        data["points_2d"],
        data["confidences"],
        data["joints_3d"],
        K,
        R,
        t,
        clip_len=args.clip_len,
        device=device,
        min_views=args.min_views,
        max_views=args.max_views,
        num_subsets_per_k=args.num_subsets_per_k,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved variable-view results to {out_path}")

    print(f"{'k':>3} | MPJPE (mm) | std | subsets")
    for k, r in results.items():
        print(f"{k:>3} | {r['mean_mm']:.2f} | {r['std_mm']:.2f} | {r['n_subsets']}")


if __name__ == "__main__":
    main()
