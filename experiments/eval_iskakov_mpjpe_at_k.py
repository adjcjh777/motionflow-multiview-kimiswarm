#!/usr/bin/env python3
"""MPJPE@k sparse-view curves for the Iskakov learnable-triangulation baseline.

Evaluates a trained ``IskakovLearnableTriangulation`` checkpoint on subsets of
k views (k = 2..V) with deterministic subset sampling, alongside the frozen
unweighted / confidence-weighted DLT references, so the sparse-view robustness
claim can be plotted. Metrics: direct MPJPE and root-aligned MPJPE (mm),
macro-mean over the eval files.

Protocol anchors:
* H36M true-GT (issue #194): 4 views -> k = 2, 3, 4; val S9/S11 full test.
* Shelf/Campus detected (Campus primary): 3 views -> k = 2, 3.

Usage:
    python experiments/eval_iskakov_mpjpe_at_k.py \
        --protocol h36m \
        --checkpoint outputs/iskakov_learnable_tri_h36m_true_gt.pth \
        --num_subsets 10 --output_dir outputs/iskakov_mpjpe_at_k
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.iskakov_learnable_triangulation import (  # noqa: E402
    IskakovLearnableTriangulation,
    build_projection_matrices,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq  # noqa: E402

H36M_VAL = [
    "data/h36m_true_gt/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
]
H36M_TRUE_GT_V2_VAL = [
    "data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
    "data/h36m_true_gt_v2/s_11_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz",
]
CAMPUS_VAL = "data/webbridge/shelf_campus_detected/campus_seq1_val_detected_m.npz"
SHELF_VAL = "data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=["h36m", "h36m_true_gt_v2", "shelf_campus"], default="h36m")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--per_view", action="store_true")
    p.add_argument("--num_subsets", type=int, default=10,
                   help="random view subsets per k (0 = enumerate all subsets)")
    p.add_argument("--max_frames", type=int, default=4000,
                   help="deterministic stride cap per eval file (memory)")
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--output_dir", default="outputs/iskakov_mpjpe_at_k")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def direct_mm(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - gt, dim=-1).mean() * 1000.0)


def root_mm(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pc = pred - pred.mean(dim=-2, keepdim=True)
    gc = gt - gt.mean(dim=-2, keepdim=True)
    return direct_mm(pc, gc)


@torch.no_grad()
def eval_subset(model, p2d, conf, gt, K, R, t, views):
    sel = list(views)
    X = model(p2d[:, sel], conf[:, sel], K[sel], R[sel], t[sel])
    return direct_mm(X, gt), root_mm(X, gt)


@torch.no_grad()
def frozen_dlt_subset(p2d, conf, gt, K, R, t, views):
    """Unweighted + conf-weighted DLT for one view subset (float64, SVD-based)."""
    from motionflow_mv.fusion.triangulation import triangulate_dlt

    sel = list(views)
    P = build_projection_matrices(K, R, t).cpu().numpy()[sel]
    p = p2d[:, sel].cpu().numpy().astype(np.float64)
    c = conf[:, sel].cpu().numpy().astype(np.float64)
    g = gt.cpu().numpy().astype(np.float64)
    N, V, J, _ = p.shape
    re_uw = np.zeros_like(g)
    re_cw = np.zeros_like(g)
    for f in range(N):
        for j in range(J):
            re_uw[f, j] = triangulate_dlt(p[f, :, j], P)
            re_cw[f, j] = triangulate_dlt(p[f, :, j], P, weights=c[f, :, j])
    return {
        "unweighted_direct": float(np.linalg.norm(re_uw - g, axis=-1).mean() * 1000),
        "unweighted_root": float(
            np.linalg.norm((re_uw - re_uw.mean(-2, keepdims=True))
                           - (g - g.mean(-2, keepdims=True)), axis=-1).mean() * 1000),
        "conf_direct": float(np.linalg.norm(re_cw - g, axis=-1).mean() * 1000),
        "conf_root": float(
            np.linalg.norm((re_cw - re_cw.mean(-2, keepdims=True))
                           - (g - g.mean(-2, keepdims=True)), axis=-1).mean() * 1000),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.protocol == "h36m":
        files = [(Path(p).stem.split("_acts_")[0], p) for p in H36M_VAL]
        k_values = [2, 3, 4]
    elif args.protocol == "h36m_true_gt_v2":
        files = [(Path(p).stem.split("_acts_")[0], p) for p in H36M_TRUE_GT_V2_VAL]
        k_values = [2, 3, 4]
    else:
        files = [("campus", CAMPUS_VAL), ("shelf", SHELF_VAL)]
        k_values = [2, 3, 4, 5]  # Shelf has 5 views; Campus skips k>3.

    # Load checkpoint weights.
    state = torch.load(args.checkpoint, map_location="cpu")["model_state"]

    results = {"protocol": args.protocol, "checkpoint": args.checkpoint,
               "k_values": k_values, "per_file": {}, "macro": {}}
    for name, path in files:
        d = np.load(path)
        T = d["points_2d"].shape[0]
        stride = max(1, T // args.max_frames)
        idx = np.arange(0, T, stride)
        p2d = torch.from_numpy(d["points_2d"][idx]).float().to(device)
        conf = torch.from_numpy(d["confidences"][idx]).float().to(device)
        gt = torch.from_numpy(d["joints_3d"][idx]).float().to(device)
        K = torch.from_numpy(d["camera_K"]).float().to(device)
        R = torch.from_numpy(d["camera_R"]).float().to(device)
        t = torch.from_numpy(d["camera_t"]).float().to(device)
        V = p2d.shape[1]

        model = IskakovLearnableTriangulation(
            hidden_dim=args.hidden_dim, cross_view=not args.per_view
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        per_k = {}
        for k in k_values:
            if k > V:
                continue
            if args.num_subsets == 0 or math.comb(V, k) <= args.num_subsets:
                subsets = list(itertools.combinations(range(V), k))
            else:
                subsets = []
                while len(subsets) < args.num_subsets:
                    s = tuple(sorted(rng.choice(V, size=k, replace=False).tolist()))
                    if s not in subsets:
                        subsets.append(s)
            dirs, roots = [], []
            for s in subsets:
                dv, rv = eval_subset(model, p2d, conf, gt, K, R, t, s)
                dirs.append(dv); roots.append(rv)
            # Frozen refs only on the first subset per k (SVD loop is slow;
            # unweighted/conf DLT depend weakly on which views are chosen, so
            # we also report the exact first-subset numbers).
            fr = frozen_dlt_subset(p2d, conf, gt, K.cpu(), R.cpu(), t.cpu(), subsets[0])
            per_k[k] = {
                "n_subsets": len(subsets),
                "learned_direct_mm": float(np.mean(dirs)),
                "learned_root_mm": float(np.mean(roots)),
                "learned_direct_mm_std": float(np.std(dirs)),
                "frozen_first_subset": fr,
            }
            print(f"[{name}] k={k}: learned direct {per_k[k]['learned_direct_mm']:.2f} "
                  f"(+-{per_k[k]['learned_direct_mm_std']:.2f}) root {per_k[k]['learned_root_mm']:.2f} mm; "
                  f"DLT(uw/conf) first subset {fr['unweighted_direct']:.2f}/{fr['conf_direct']:.2f} mm",
                  flush=True)
        results["per_file"][name] = {"T_eval": int(len(idx)), "V": int(V), "k": per_k}

    # Macro mean over files.
    for k in k_values:
        rows = [f["k"][k] for f in results["per_file"].values() if k in f["k"]]
        if rows:
            results["macro"][k] = {
                "learned_direct_mm": float(np.mean([r["learned_direct_mm"] for r in rows])),
                "learned_root_mm": float(np.mean([r["learned_root_mm"] for r in rows])),
            }
            print(f"[macro] k={k}: direct {results['macro'][k]['learned_direct_mm']:.2f} "
                  f"root {results['macro'][k]['learned_root_mm']:.2f} mm", flush=True)

    out = Path(args.output_dir) / f"iskakov_mpjpe_at_k_{args.protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
