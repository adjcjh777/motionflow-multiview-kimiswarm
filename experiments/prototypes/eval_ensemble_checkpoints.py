"""Evaluate an multi-checkpoint ensemble of any registered model.

Example
-------
    python experiments/prototypes/eval_ensemble_checkpoints.py \
        --model bayesian_tri_v2_pp \
        --dataset data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp.pth \
        --checkpoint outputs/bayesian_tri_v2_large_scale_mpiinf3dhp_epoch45.pth \
        --clip_len 13 --d 128 --residual_hidden 256 --n_st_layers 3 \
        --val_stride 50 \
        --output_json outputs/bayesian_tri_v2_ensemble_eval.json

The script accepts every argument that ``experiments/eval_full_metrics.py``
accepts, plus a repeated ``--checkpoint`` argument.  Predictions from every
loaded checkpoint are averaged (optionally weighted) before metrics are
computed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from motionflow_mv.eval.metrics import compute_all_metrics, summarize_metrics
from motionflow_mv.fusion.prototypes.ensemble_predictor import MultiCheckpointEnsemble
from experiments.eval_full_metrics import (
    MODEL_CLASSES,
    TemporalClipDataset,
    build_model,
    collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MODEL_CLASSES), required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Path to a model checkpoint; can be repeated.",
    )
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n_temporal_layers", type=int, default=2)
    parser.add_argument("--n_st_layers", type=int, default=2)
    parser.add_argument("--n_view_layers", type=int, default=2)
    parser.add_argument("--n_view_groups", type=int, default=2)
    parser.add_argument("--n_joint_graph_layers", type=int, default=1)
    parser.add_argument("--no_skeleton_graph", action="store_true")
    parser.add_argument("--residual_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--target_k", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--gt_scale", type=float, default=1.0)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--parents", type=str, default=None)
    parser.add_argument("--symmetry_pairs", type=str, default=None)
    parser.add_argument("--source_n_views", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Optional per-checkpoint weights for weighted averaging.")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    n_views = data["camera_K"].shape[0]
    j = data["points_2d"].shape[2]

    dataset = TemporalClipDataset(
        args.dataset,
        args.clip_len,
        stride=args.val_stride,
        gt_scale=args.gt_scale,
        camera_scale=args.camera_scale,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    source_n_views = args.source_n_views if args.source_n_views is not None else n_views

    # build_fn must produce a model matching the requested architecture.
    def build_fn():
        return build_model(args, source_n_views, j)

    ensemble = MultiCheckpointEnsemble(
        build_fn,
        args.checkpoint,
        device=device,
        weights=args.weights,
    )

    preds, gts = [], []
    with torch.no_grad():
        for xb, yb, K, R, t in loader:
            xb, yb = xb.to(device), yb.to(device)
            K, R, t = K.to(device), R.to(device), t.to(device)
            pred = ensemble(xb, K=K, R=R, t=t)
            preds.append(pred.cpu().numpy())
            gts.append(yb.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], 3)
    gts = gts.reshape(-1, gts.shape[-2], 3)

    parents_arr = None
    if args.parents:
        from experiments.eval_full_metrics import _load_list
        parents_arr = np.array(_load_list(args.parents, int), dtype=np.int64)

    report = compute_all_metrics(preds * 1000.0, gts * 1000.0, parents=parents_arr)
    print(summarize_metrics(report))
    print(f"MPJPE: {report['mpjpe']:.2f} mm")
    print(f"PA-MPJPE: {report['pa_mpjpe']:.2f} mm")
    print(f"PCK@50: {report['pck@50mm']:.4f}")
    print(f"PCK@100: {report['pck@100mm']:.4f}")
    print(f"PCK@150: {report['pck@150mm']:.4f}")
    print(f"PCK-AUC: {report['pck_auc']:.4f}")

    if args.output_json:
        serializable = {}
        for k, v in report.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = float(v)
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
