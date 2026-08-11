#!/usr/bin/env python3
"""Train the Iskakov et al. (ICCV 2019) Learnable Triangulation baseline on the
non-circular true-GT protocols.

Reference: Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y., 'Learnable
Triangulation of Human Pose', ICCV 2019, arXiv:1905.05754. (Re-implementation
of the weight-prediction branch on raw detected 2D features; see
motionflow_mv/fusion/iskakov_learnable_triangulation.py.)

Protocols
---------
* ``--protocol shelf_campus`` (default): the true-GT detected Shelf/Campus
  protocol of docs/results_true_gt_shelf_campus.md
  (manifest configs/splits/shelf_campus_detected_smoke.yaml).
* ``--protocol h36m``: the H36M true-GT standard protocol
  (S1,5,6,7,8 train -> S9/S11 test; manifest
  configs/splits/h36m_true_gt_standard.yaml; issue #194).  Labels come from
  the official-mocap-derived data_3d_h36m.npz, NOT from DLT triangulation.

Loss is MPJPE in mm on metres; root-aligned (centroid-aligned) MPJPE is logged
as a secondary metric.  Frozen references on val (unweighted DLT and
confidence-weighted DLT, same SVD routine as the leaderboard diagnostics) are
computed on a stride-subsampled val subset for the large H36M protocol
(--ref_max_frames, deterministic).

Usage:
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --datasets shelf+campus --epochs 30 --log_path outputs/iskakov_learnable_tri_detected.log
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --datasets campus --epochs 30 --log_path outputs/iskakov_learnable_tri_campus_only.log
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --protocol h36m --epochs 10 --train_samples_per_epoch 4096 \
      --log_path outputs/iskakov_learnable_tri_h36m_true_gt.log
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --protocol aist_smoke --epochs 30 --batch_size 4 --train_samples_per_epoch 128 \
      --log_path outputs/iskakov_learnable_tri_aist_only_smoke.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.iskakov_learnable_triangulation import (
    IskakovLearnableTriangulation,
    build_projection_matrices,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt

DATASET_KEYS = {"shelf", "campus", "h36m", "aist"}

SHELF_CAMPUS_FILES = {
    "shelf": (
        "data/webbridge/shelf_campus_detected/shelf_seq1_train_detected_m.npz",
        "data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz",
    ),
    "campus": (
        "data/webbridge/shelf_campus_detected/campus_seq1_train_detected_m.npz",
        "data/webbridge/shelf_campus_detected/campus_seq1_val_detected_m.npz",
    ),
}

H36M_TRAIN = [1, 5, 6, 7, 8]
H36M_TEST = [9, 11]

# AIST++ smoke split: ch01 + ch02 train, ch03 val (9 views, 17 joints, H36M skeleton).
AIST_SMOKE_FILES = {
    "aist_ch01": (
        "data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch01_multiview.npz",
        "data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch03_multiview.npz",
    ),
    "aist_ch02": (
        "data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch02_multiview.npz",
        "data/webbridge/aistpp_canonical/gBR_sBM_cAll_d04_mBR0_ch03_multiview.npz",
    ),
}


def h36m_files(subject: int) -> Tuple[str, str]:
    stem = f"data/h36m_true_gt/s_{subject:02d}_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz"
    return stem, stem


PROTOCOL_MANIFESTS = {
    "shelf_campus": "configs/splits/shelf_campus_detected_smoke.yaml",
    "h36m": "configs/splits/h36m_true_gt_standard.yaml",
    "aist_smoke": "configs/splits/aist_only_smoke.yaml",
}


def build_dataset_files(protocol: str) -> Dict[str, Tuple[str, str]]:
    if protocol == "shelf_campus":
        return dict(SHELF_CAMPUS_FILES)
    if protocol == "h36m":
        out = {}
        for s in H36M_TRAIN + H36M_TEST:
            out[f"h36m_s{s}"] = h36m_files(s)
        return out
    if protocol == "aist_smoke":
        return dict(AIST_SMOKE_FILES)
    raise ValueError(f"unknown protocol {protocol!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="shelf_campus", choices=["shelf_campus", "h36m", "aist_smoke"])
    p.add_argument("--manifest", default=None,
                   help="defaults to the protocol's canonical manifest")
    p.add_argument("--datasets", default="shelf+campus",
                   help="'shelf+campus' / 'campus' / 'shelf' for the shelf_campus "
                        "protocol; ignored for h36m (all 5 train + 2 test subjects)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="early-stop patience on combined direct val MPJPE")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--per_view", action="store_true", help="disable cross-view feature mode (ablation)")
    p.add_argument("--view_dropout_prob", type=float, default=0.0,
                   help="per-step probability of applying random view dropout during training")
    p.add_argument("--view_dropout_min_views", type=int, default=2,
                   help="minimum number of kept views when view dropout fires")
    p.add_argument("--view_dropout_mode", choices=["subset", "mask"], default="subset",
                   help="subset: physically drop views (features + solve) — matches eval-time "
                        "sparse subsets; mask: keep all features, zero the DLT weights of "
                        "dropped views")
    p.add_argument("--train_samples_per_epoch", type=int, default=0,
                   help="if >0, sample this many frames per epoch instead of a full pass")
    p.add_argument("--ref_max_frames", type=int, default=2000,
                   help="max val frames used for the frozen DLT references (deterministic stride)")
    p.add_argument("--log_path", default=None)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_manifest(path: str, dataset_files: Dict[str, Tuple[str, str]]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    # Verify manifest agrees with the canonical npz paths used here.
    want = set()
    for tr, va in dataset_files.values():
        want.add(tr)
        want.add(va)
    listed = set(m.get("train_paths", [])) | set(m.get("val_paths", []))
    if not want.issubset(listed):
        raise RuntimeError(f"manifest {path} missing expected paths: {want - listed}")
    return m


class FrameDataset:
    """One split as GPU-ready tensors."""

    def __init__(self, npz_path: str, device: torch.device):
        d = np.load(npz_path)
        self.path = npz_path
        self.points_2d = torch.from_numpy(d["points_2d"]).float().to(device)   # (N, V, J, 2)
        self.confidences = torch.from_numpy(d["confidences"]).float().to(device)  # (N, V, J)
        self.joints_3d = torch.from_numpy(d["joints_3d"]).float().to(device)  # (N, J, 3)
        self.K = torch.from_numpy(d["camera_K"]).float().to(device)
        self.R = torch.from_numpy(d["camera_R"]).float().to(device)
        self.t = torch.from_numpy(d["camera_t"]).float().to(device)
        self.n = self.points_2d.shape[0]

    def batch(self, idx: torch.Tensor):
        return (
            self.points_2d[idx], self.confidences[idx], self.joints_3d[idx],
            self.K, self.R, self.t,
        )


def mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean per-joint position error in mm (inputs in metres)."""
    return torch.linalg.norm(pred - gt, dim=-1).mean() * 1000.0


def centroid_aligned_mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Centroid-aligned MPJPE in mm — identical definition to
    scripts/diagnose_circular_labels.py so numbers match the leaderboard doc."""
    pc = pred - pred.mean(dim=-2, keepdim=True)
    gc = gt - gt.mean(dim=-2, keepdim=True)
    return mpjpe_mm(pc, gc)


@torch.no_grad()
def evaluate(model, datasets: Dict[str, FrameDataset]) -> dict:
    """Eval direct + centroid-aligned MPJPE per dataset and their macro mean."""
    out = {}
    for name, ds in datasets.items():
        X = model(ds.points_2d, ds.confidences, ds.K, ds.R, ds.t)
        out[f"{name}_direct_mm"] = float(mpjpe_mm(X, ds.joints_3d))
        out[f"{name}_root_mm"] = float(centroid_aligned_mpjpe_mm(X, ds.joints_3d))
    out["combined_direct_mm"] = float(np.mean([out[f"{n}_direct_mm"] for n in datasets]))
    out["combined_root_mm"] = float(np.mean([out[f"{n}_root_mm"] for n in datasets]))
    return out


@torch.no_grad()
def frozen_dlt_references(
    val_datasets: Dict[str, FrameDataset], ref_max_frames: int = 2000
) -> dict:
    """Unweighted and confidence-weighted DLT on val using the leaderboard's
    SVD routine (triangulate_dlt), per dataset + macro mean.

    Large val sets are stride-subsampled to *ref_max_frames* frames
    (deterministic) because the per-frame SVD loop is O(N*J).
    """
    refs: Dict[str, Dict[str, float]] = {}
    for name, ds in val_datasets.items():
        P = build_projection_matrices(ds.K.cpu(), ds.R.cpu(), ds.t.cpu()).double().numpy()
        N = ds.points_2d.shape[0]
        stride = max(1, int(math.ceil(N / ref_max_frames)))
        idx = np.arange(0, N, stride)
        p2d = ds.points_2d[idx].double().cpu().numpy()
        conf = ds.confidences[idx].double().cpu().numpy()
        gt = ds.joints_3d[idx].double().cpu().numpy()
        _, V, J, _ = p2d.shape
        re_uw = np.zeros_like(gt)
        re_cw = np.zeros_like(gt)
        for f in range(p2d.shape[0]):
            for j in range(J):
                re_uw[f, j] = triangulate_dlt(p2d[f, :, j], P)
                re_cw[f, j] = triangulate_dlt(p2d[f, :, j], P, weights=conf[f, :, j])

        def direct(p): return float(np.linalg.norm(p - gt, axis=-1).mean() * 1000)

        def root(p):
            pc = p - p.mean(axis=-2, keepdims=True)
            gc = gt - gt.mean(axis=-2, keepdims=True)
            return float(np.linalg.norm(pc - gc, axis=-1).mean() * 1000)

        refs[name] = {
            "unweighted_direct_mm": direct(re_uw), "unweighted_root_mm": root(re_uw),
            "conf_direct_mm": direct(re_cw), "conf_root_mm": root(re_cw),
            "n_ref_frames": int(len(idx)),
        }
    out: Dict[str, float] = {}
    for key in ("unweighted_direct_mm", "unweighted_root_mm", "conf_direct_mm", "conf_root_mm"):
        out[f"combined_{key}"] = float(np.mean([refs[n][key] for n in val_datasets]))
        for n in val_datasets:
            out[f"{n}_{key}"] = refs[n][key]
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    protocol = args.protocol
    dataset_files = build_dataset_files(protocol)
    manifest_path = args.manifest or PROTOCOL_MANIFESTS[protocol]

    log_path = Path(args.log_path or {
        "shelf_campus": "outputs/iskakov_learnable_tri_detected.log",
        "h36m": "outputs/iskakov_learnable_tri_h36m_true_gt.log",
        "aist_smoke": "outputs/iskakov_learnable_tri_aist_only_smoke.log",
    }[protocol])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    log("=== Iskakov Learnable Triangulation baseline (ICCV 2019 re-implementation) ===")
    log(f"protocol={protocol} device={device} torch={torch.__version__} seed={args.seed}")
    log(f"manifest={manifest_path}")

    manifest = load_manifest(manifest_path, dataset_files)
    log(f"manifest name={manifest.get('name')}")

    if protocol == "shelf_campus":
        wanted = [s.strip() for s in args.datasets.split("+")]
        for w in wanted:
            if w not in SHELF_CAMPUS_FILES:
                raise SystemExit(f"unknown dataset '{w}'")
        train_names = wanted
        val_names = wanted
    elif protocol == "h36m":
        train_names = [f"h36m_s{s}" for s in H36M_TRAIN]
        val_names = [f"h36m_s{s}" for s in H36M_TEST]
    else:  # aist_smoke
        # Train on ch01 + ch02; val on ch03. Evaluate ch03 only once to avoid
        # double-counting in the combined metric.
        train_names = ["aist_ch01", "aist_ch02"]
        val_names = ["aist_ch01"]

    train_ds = {w: FrameDataset(dataset_files[w][0], device) for w in train_names}
    val_ds = {w: FrameDataset(dataset_files[w][1], device) for w in val_names}
    for w in train_names:
        log(f"train {w}: {train_ds[w].n} frames, views={train_ds[w].points_2d.shape[1]}")
    for w in val_names:
        log(f"val   {w}: {val_ds[w].n} frames, views={val_ds[w].points_2d.shape[1]}")

    cross_view = not args.per_view
    model = IskakovLearnableTriangulation(hidden_dim=args.hidden_dim, cross_view=cross_view).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: IskakovLearnableTriangulation hidden_dim={args.hidden_dim} "
        f"cross_view={cross_view} params={n_params} "
        f"view_dropout_prob={args.view_dropout_prob} min_views={args.view_dropout_min_views}")

    # ---- Frozen references (no learning) -----------------------------------
    log(f"computing frozen DLT references on val (SVD routine, "
        f"ref_max_frames={args.ref_max_frames}, deterministic stride)...")
    refs = frozen_dlt_references(val_ds, ref_max_frames=args.ref_max_frames)
    for k, v in sorted(refs.items()):
        if not k.endswith("n_ref_frames"):
            log(f"FROZEN {k} = {v:.2f} mm")

    # ---- Training ----------------------------------------------------------
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.train_samples_per_epoch > 0:
        samples_per_epoch = args.train_samples_per_epoch
    else:
        samples_per_epoch = max(ds.n for ds in train_ds.values())
    total_steps = args.epochs * samples_per_epoch // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    best = {"combined_direct_mm": float("inf"), "epoch": -1, "metrics": None}
    epochs_no_improve = 0
    ckpt_path = Path(args.ckpt_path or log_path.with_suffix(".pth"))
    history = []
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Domain-balanced: each step picks a domain uniformly, then a random
        # batch from that dataset (keeps view count constant within a batch).
        epoch_loss = 0.0
        epoch_root = 0.0
        n_steps = 0
        n_batches = max(1, samples_per_epoch // args.batch_size)
        for b in range(n_batches):
            w = train_names[int(rng.integers(len(train_names)))]
            ds = train_ds[w]
            idx = torch.from_numpy(rng.integers(0, ds.n, size=args.batch_size)).long().to(device)

            p2d, conf, gt, K, R, t = ds.batch(idx)
            view_mask = None
            if args.view_dropout_prob > 0 and rng.random() < args.view_dropout_prob:
                V = p2d.shape[1]
                lo = max(args.view_dropout_min_views, 1)
                k = int(rng.integers(lo, V + 1)) if V > lo else V
                keep = np.sort(rng.choice(V, size=k, replace=False))
                if args.view_dropout_mode == "subset" and k < V:
                    # Physically drop views so training matches eval-time
                    # sparse-view subsets (MPJPE@k protocol).
                    keep_t = torch.from_numpy(keep).long().to(device)
                    p2d = p2d[:, keep_t]
                    conf = conf[:, keep_t]
                    K = K[keep_t]
                    R = R[keep_t]
                    t = t[keep_t]
                elif k < V:
                    view_mask = torch.zeros(p2d.shape[0], V, device=device)
                    view_mask[:, keep] = 1.0
            pred = model(p2d, conf, K, R, t, view_mask=view_mask)
            loss = mpjpe_mm(pred, gt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/inf loss at epoch {epoch} step {b}")
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            step += 1
            n_steps += 1
            epoch_loss += float(loss)
            epoch_root += float(centroid_aligned_mpjpe_mm(pred, gt))

        model.eval()
        val = evaluate(model, val_ds)
        train_mm = epoch_loss / n_steps
        train_root_mm = epoch_root / n_steps
        lr_now = optim.param_groups[0]["lr"]
        parts = [f"epoch {epoch:03d}/{args.epochs}", f"lr={lr_now:.2e}",
                 f"train_MPJPE={train_mm:.2f} mm", f"train_root={train_root_mm:.2f} mm"]
        for w in val_names:
            parts.append(f"val[{w}] direct={val[f'{w}_direct_mm']:.2f} root={val[f'{w}_root_mm']:.2f} mm")
        parts.append(f"val[combined] direct={val['combined_direct_mm']:.2f} root={val['combined_root_mm']:.2f} mm")
        log(" | ".join(parts))
        history.append({"epoch": epoch, "train_direct_mm": train_mm, **val})

        for m in val.values():
            if not math.isfinite(m):
                raise RuntimeError(f"NaN/inf val metric at epoch {epoch}")

        if val["combined_direct_mm"] < best["combined_direct_mm"] - 1e-9:
            best = {"combined_direct_mm": val["combined_direct_mm"], "epoch": epoch, "metrics": val}
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)
            epochs_no_improve = 0
            log(f"  new best val direct MPJPE = {val['combined_direct_mm']:.2f} mm (ckpt {ckpt_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log(f"early stop at epoch {epoch} (patience {args.patience})")
                break

    # ---- Summary ------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    final_val = evaluate(model, val_ds)
    log("=== SUMMARY ===")
    log(f"best epoch = {best['epoch']}")
    for k, v in sorted(final_val.items()):
        log(f"FINAL val {k} = {v:.2f} mm")
    for k, v in sorted(refs.items()):
        if not k.endswith("n_ref_frames"):
            log(f"FROZEN ref {k} = {v:.2f} mm")
    gain = refs.get("combined_conf_direct_mm", float("nan")) - final_val["combined_direct_mm"]
    log(f"learned-vs-confDLT gain (combined direct) = {gain:+.2f} mm")
    log(f"learned-vs-unweightedDLT gain (combined direct) = "
        f"{refs.get('combined_unweighted_direct_mm', float('nan')) - final_val['combined_direct_mm']:+.2f} mm")
    log(f"checkpoint: {ckpt_path} | log: {log_path}")

    cfg_path = log_path.with_suffix(".config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_params": n_params, "best": best,
                   "frozen_references": refs, "history": history}, f, indent=2)
    log(f"config/history: {cfg_path}")
    log_f.close()


if __name__ == "__main__":
    main()
