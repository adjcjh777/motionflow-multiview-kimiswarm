"""Iterative residual refinement evaluation for RayAttentionFusionModelTemporalResidual.

Loads a trained residual checkpoint and evaluates it on an MPI-INF-3DHP
validation sequence with ``n_iter = 1, 2, 3`` residual-head applications.
The per-view attention weights and DLT triangulation are computed once;
the residual MLP is then applied repeatedly, feeding the updated 3D
pose back into the head.

Usage
-----
    conda run -n mf python experiments/eval_ray_attention_temporal_residual_iterative.py \
        --checkpoint outputs/ray_attention_temporal_residual_v2.pth \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --batch_size 8 --out docs/swarm_iter6/iterative_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics
from motionflow_mv.fusion.ray_attention_temporal_residual_model import RayAttentionFusionModelTemporalResidual
from train_ray_attention_temporal_residual_mpiinf3dhp import TemporalClipDataset, collate_fn


def evaluate_n_iter(model, loader, device, n_iter):
    model.eval()
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)

            pred, _ = model(xb, K=K, R=R, t=t, n_iter=n_iter)  # (B, T, J, 3)
            all_pred.append(pred.cpu().numpy())
            all_gt.append(yb.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)  # (N_clips, T, J, 3)
    all_gt = np.concatenate(all_gt, axis=0)
    n_clips, T, J, _ = all_pred.shape
    pred_flat = all_pred.reshape(-1, J, 3) * 1000.0  # convert m -> mm
    gt_flat = all_gt.reshape(-1, J, 3) * 1000.0

    report = compute_all_metrics(pred_flat, gt_flat)
    return {
        "n_iter": n_iter,
        "n_clips": int(n_clips),
        "clip_len": int(T),
        "n_joints": int(J),
        "mpjpe_mm": float(report["mpjpe"]),
        "pa_mpjpe_mm": float(report["pa_mpjpe"]),
        "pck_50mm": float(report.get("pck@50mm", None)),
        "pck_100mm": float(report.get("pck@100mm", None)),
        "pck_150mm": float(report.get("pck@150mm", None)),
        "pck_auc_150mm": float(report["pck_auc"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate iterative residual refinement on MPI-INF-3DHP"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--val", type=str, required=True, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_n_iter", type=int, default=3, help="Maximum number of residual iterations to evaluate")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON path to write results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Infer dimensions from data.
    data = np.load(args.val)
    n_views = int(data["camera_K"].shape[0])
    j = int(data["points_2d"].shape[2])
    print(f"Data: {args.val}")
    print(f"n_views={n_views}, n_joints={j}, clip_len={args.clip_len}, d={args.d}")

    val_dataset = TemporalClipDataset(args.val, args.clip_len, stride=1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = RayAttentionFusionModelTemporalResidual(
        j=j, d=args.d, n_views=n_views, n_temporal_layers=args.n_temporal_layers,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    results = []
    for n_iter in range(1, args.max_n_iter + 1):
        print(f"\n--- Evaluating n_iter={n_iter} ---")
        summary = evaluate_n_iter(model, val_loader, device, n_iter)
        results.append(summary)
        print(f"n_iter={n_iter}: MPJPE={summary['mpjpe_mm']:.4f} mm, "
              f"PA-MPJPE={summary['pa_mpjpe_mm']:.4f} mm, "
              f"PCK@50mm={summary['pck_50mm']:.4f}, "
              f"PCK@100mm={summary['pck_100mm']:.4f}, "
              f"PCK@150mm={summary['pck_150mm']:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "val_dataset": str(args.val),
                "n_views": n_views,
                "d": args.d,
                "n_temporal_layers": args.n_temporal_layers,
                "results": results,
            }, f, indent=2)
        print(f"\nSaved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
