#!/usr/bin/env python3
"""Train the Iskakov et al. (ICCV 2019) Learnable Triangulation baseline on the
non-circular detected Shelf/Campus protocol.

Reference: Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y., 'Learnable
Triangulation of Human Pose', ICCV 2019, arXiv:1905.05754. (Re-implementation of the weight-prediction branch on raw detected
2D features; see motionflow_mv/fusion/iskakov_learnable_triangulation.py.)

Protocol follows docs/results_true_gt_shelf_campus.md:
  * manifest: configs/splits/shelf_campus_detected_smoke.yaml
  * loss: MPJPE in mm on metres (root-aligned/centroid-aligned logged as secondary)
  * frozen references on val: unweighted DLT and confidence-weighted DLT
    (same SVD routine as the leaderboard diagnostic)

Usage:
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --datasets shelf+campus --epochs 30 --log_path outputs/iskakov_learnable_tri_detected.log
  python experiments/train_iskakov_baseline_shelf_campus.py \
      --datasets campus --epochs 30 --log_path outputs/iskakov_learnable_tri_campus_only.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

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

DATASET_KEYS = {"shelf", "campus"}
DATASET_FILES = {
    "shelf": (
        "data/webbridge/shelf_campus_detected/shelf_seq1_train_detected_m.npz",
        "data/webbridge/shelf_campus_detected/shelf_seq1_val_detected_m.npz",
    ),
    "campus": (
        "data/webbridge/shelf_campus_detected/campus_seq1_train_detected_m.npz",
        "data/webbridge/shelf_campus_detected/campus_seq1_val_detected_m.npz",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/splits/shelf_campus_detected_smoke.yaml")
    p.add_argument("--datasets", default="shelf+campus",
                   help="'shelf+campus' (domain-balanced), 'campus', or 'shelf'")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8, help="early-stop patience on combined direct val MPJPE")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--per_view", action="store_true", help="disable cross-view feature mode (ablation)")
    p.add_argument("--log_path", default="outputs/iskakov_learnable_tri_detected.log")
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    # Verify manifest agrees with the canonical npz paths used here.
    want = set()
    for tr, va in DATASET_FILES.values():
        want.add(tr)
        want.add(va)
    listed = set(m.get("train_paths", [])) | set(m.get("val_paths", []))
    if not want.issubset(listed):
        raise RuntimeError(f"manifest {path} missing expected paths: {want - listed}")
    return m


class FrameDataset:
    """One Shelf or Campus split as GPU-ready tensors."""

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
def evaluate(model, datasets: dict[str, FrameDataset]) -> dict:
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
def frozen_dlt_references(val_datasets: dict[str, FrameDataset]) -> dict:
    """Unweighted and confidence-weighted DLT on val using the leaderboard's
    SVD routine (triangulate_dlt), per dataset + macro mean."""
    refs: dict[str, dict[str, float]] = {}
    for name, ds in val_datasets.items():
        P = build_projection_matrices(ds.K.cpu(), ds.R.cpu(), ds.t.cpu()).double().numpy()
        p2d = ds.points_2d.double().cpu().numpy()
        conf = ds.confidences.double().cpu().numpy()
        gt = ds.joints_3d.double().cpu().numpy()
        N, V, J, _ = p2d.shape
        re_uw = np.zeros_like(gt)
        re_cw = np.zeros_like(gt)
        for f in range(N):
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
        }
    out: dict[str, float] = {}
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

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    log(f"=== Iskakov Learnable Triangulation baseline (ICCV 2019 re-implementation) ===")
    log(f"device={device} torch={torch.__version__} seed={args.seed}")
    log(f"manifest={args.manifest} datasets={args.datasets}")

    manifest = load_manifest(args.manifest)
    log(f"manifest name={manifest.get('name')}")

    wanted = [s.strip() for s in args.datasets.split("+")]
    for w in wanted:
        if w not in DATASET_KEYS:
            raise SystemExit(f"unknown dataset '{w}'")

    train_ds = {w: FrameDataset(DATASET_FILES[w][0], device) for w in wanted}
    val_ds = {w: FrameDataset(DATASET_FILES[w][1], device) for w in wanted}
    for w in wanted:
        log(f"loaded {w}: train={train_ds[w].n} frames, val={val_ds[w].n} frames, "
            f"views={train_ds[w].points_2d.shape[1]}")

    cross_view = not args.per_view
    model = IskakovLearnableTriangulation(hidden_dim=args.hidden_dim, cross_view=cross_view).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: IskakovLearnableTriangulation hidden_dim={args.hidden_dim} "
        f"cross_view={cross_view} params={n_params}")

    # ---- Frozen references (no learning) -----------------------------------
    log("computing frozen DLT references on val (SVD routine, leaderboard-consistent)...")
    refs = frozen_dlt_references(val_ds)
    for k, v in sorted(refs.items()):
        log(f"FROZEN {k} = {v:.2f} mm")

    # ---- Training ----------------------------------------------------------
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(ds.n for ds in train_ds.values()) // args.batch_size
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
        order = {w: rng.permutation(train_ds[w].n) for w in wanted}
        cursors = {w: 0 for w in wanted}
        n_batches_max = max(math.ceil(ds.n / args.batch_size) for ds in train_ds.values())
        epoch_loss = 0.0
        epoch_root = 0.0
        n_steps = 0
        for b in range(n_batches_max):
            w = wanted[int(rng.integers(len(wanted)))]
            ds = train_ds[w]
            start = cursors[w]
            if start >= ds.n:  # wrap: domain stays balanced across the wrap
                order[w] = rng.permutation(ds.n)
                start = 0
            idx = torch.from_numpy(order[w][start:start + args.batch_size]).long().to(device)
            cursors[w] = start + args.batch_size

            p2d, conf, gt, K, R, t = ds.batch(idx)
            pred = model(p2d, conf, K, R, t)
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
        for w in wanted:
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
