#!/usr/bin/env python3
"""Train Iskakov et al. (ICCV 2019) on the full AIST++ train/val split.

Reads the canonical split from ``configs/splits/webbridge_aistpp_train_val.yaml``,
trains the learnable-triangulation baseline on 1280 train clips, and evaluates
on the full 128-clip validation set.  Numbers are written to a log, a JSON
history, and a checkpoint.

Usage
    python experiments/train_iskakov_aistpp_full.py \
        --epochs 10 --batch_size 32 --train_samples_per_epoch 4096 \
        --log_path outputs/iskakov_learnable_tri_aistpp_full.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motionflow_mv.fusion.iskakov_learnable_triangulation import (
    IskakovLearnableTriangulation,
    build_projection_matrices,
)
from motionflow_mv.fusion.triangulation import triangulate_dlt_batched_lstsq

SPLIT_PATH = Path("configs/splits/webbridge_aistpp_train_val.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--train_samples_per_epoch", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_path", default="outputs/iskakov_learnable_tri_aistpp_full.log")
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--split_path", default=str(SPLIT_PATH))
    return p.parse_args()


class Clip:
    """One AIST++ clip kept in CPU memory."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.path = str(path)
        points_2d = np.asarray(d["points_2d"], dtype=np.float32)  # (T, V, J, 2)
        confidences = np.asarray(d["confidences"], dtype=np.float32)  # (T, V, J)
        # Canonical AIST++ uses NaN for missing 2D detections. Zero them and
        # set confidence to 0 so the DLT solve ignores those views.
        nan_mask = np.isnan(points_2d).any(axis=-1)  # (T, V, J)
        points_2d = np.nan_to_num(points_2d, nan=0.0)
        confidences[nan_mask] = 0.0
        self.points_2d = points_2d
        self.confidences = confidences
        self.joints_3d = np.asarray(d["joints_3d"], dtype=np.float32)  # (T, J, 3)
        self.K = np.asarray(d["camera_K"], dtype=np.float32)  # (V, 3, 3)
        self.R = np.asarray(d["camera_R"], dtype=np.float32)  # (V, 3, 3)
        self.t = np.asarray(d["camera_t"], dtype=np.float32)  # (V, 3)
        self.T = self.points_2d.shape[0]
        self.V = self.points_2d.shape[1]
        # Some AIST++ frames have missing 3D GT (NaN).  Exclude them from
        # training and metrics.
        self.valid = np.isfinite(self.joints_3d).all(axis=(-2, -1))

    def sample_frames(self, idx: np.ndarray, device: torch.device, model: nn.Module) -> Tuple[torch.Tensor, ...] | None:
        """Return tensors on ``device`` in the dtype expected by ``model``.

        Returns ``None`` if this clip has no valid frames.
        """
        if self.valid.sum() == 0:
            return None
        idx = np.asarray(idx)
        # Restrict requested indices to the valid subset so we never feed NaN GT.
        valid_idx = np.where(self.valid)[0]
        idx = np.clip(idx, 0, len(valid_idx) - 1)
        idx = valid_idx[idx]
        dtype = next(model.parameters()).dtype
        p2d = torch.from_numpy(self.points_2d[idx]).to(device).to(dtype)
        conf = torch.from_numpy(self.confidences[idx]).to(device).to(dtype)
        gt = torch.from_numpy(self.joints_3d[idx]).to(device).to(dtype)
        K = torch.from_numpy(self.K).to(device).to(dtype)
        R = torch.from_numpy(self.R).to(device).to(dtype)
        t = torch.from_numpy(self.t).to(device).to(dtype)
        return p2d, conf, gt, K, R, t


def load_split(path: Path) -> Tuple[List[Clip], List[Clip]]:
    with open(path, "r", encoding="utf-8") as f:
        split = yaml.safe_load(f)
    train_paths = [Path(p) for p in split["train"]]
    val_paths = [Path(p) for p in split["val"]]
    print(f"Loading {len(train_paths)} train clips + {len(val_paths)} val clips...")
    train_clips = [Clip(p) for p in train_paths]
    val_clips = [Clip(p) for p in val_paths]
    return train_clips, val_clips


def mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - gt, dim=-1).mean() * 1000.0


def centroid_aligned_mpjpe_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pc = pred - pred.mean(dim=-2, keepdim=True)
    gc = gt - gt.mean(dim=-2, keepdim=True)
    return mpjpe_mm(pc, gc)


@torch.no_grad()
def evaluate_clip(model: nn.Module, clip: Clip, device: torch.device) -> Dict[str, float]:
    """Evaluate direct and centroid-aligned MPJPE on one full clip."""
    dtype = next(model.parameters()).dtype
    valid = clip.valid
    if valid.sum() == 0:
        return {"direct_mm": float("nan"), "root_mm": float("nan"), "T": clip.T}
    all_pred = []
    all_gt = []
    chunk = 256
    for start in range(0, clip.T, chunk):
        end = min(start + chunk, clip.T)
        p2d = torch.from_numpy(clip.points_2d[start:end]).to(device).to(dtype)
        conf = torch.from_numpy(clip.confidences[start:end]).to(device).to(dtype)
        gt = torch.from_numpy(clip.joints_3d[start:end]).to(device).to(dtype)
        K = torch.from_numpy(clip.K).to(device).to(dtype)
        R = torch.from_numpy(clip.R).to(device).to(dtype)
        t = torch.from_numpy(clip.t).to(device).to(dtype)
        pred = model(p2d, conf, K, R, t)
        all_pred.append(pred)
        all_gt.append(gt)
    pred = torch.cat(all_pred, dim=0)
    gt = torch.cat(all_gt, dim=0)
    pred = pred[valid]
    gt = gt[valid]
    return {
        "direct_mm": float(mpjpe_mm(pred, gt)),
        "root_mm": float(centroid_aligned_mpjpe_mm(pred, gt)),
        "T": int(valid.sum()),
    }


@torch.no_grad()
def frozen_dlt_references(model: nn.Module, val_clips: List[Clip], device: torch.device) -> Dict[str, float]:
    """Unweighted + confidence-weighted batched DLT on all val clips."""
    dtype = next(model.parameters()).dtype
    per_clip_uw, per_clip_cw = [], []
    for clip in val_clips:
        if clip.valid.sum() == 0:
            continue
        K = torch.from_numpy(clip.K).to(device).to(dtype)
        R = torch.from_numpy(clip.R).to(device).to(dtype)
        t = torch.from_numpy(clip.t).to(device).to(dtype)
        P = build_projection_matrices(K, R, t)
        all_pred_uw, all_pred_cw, all_gt = [], [], []
        for start in range(0, clip.T, 256):
            end = min(start + 256, clip.T)
            p2d = torch.from_numpy(clip.points_2d[start:end]).to(device).to(dtype)
            conf = torch.from_numpy(clip.confidences[start:end]).to(device).to(dtype)
            gt = torch.from_numpy(clip.joints_3d[start:end]).to(device).to(dtype)
            pred_uw = triangulate_dlt_batched_lstsq(p2d, P)
            pred_cw = triangulate_dlt_batched_lstsq(p2d, P, weights=conf)
            all_pred_uw.append(pred_uw)
            all_pred_cw.append(pred_cw)
            all_gt.append(gt)
        pred_uw = torch.cat(all_pred_uw, dim=0)[clip.valid]
        pred_cw = torch.cat(all_pred_cw, dim=0)[clip.valid]
        gt = torch.cat(all_gt, dim=0)[clip.valid]
        per_clip_uw.append(mpjpe_mm(pred_uw, gt).item())
        per_clip_cw.append(mpjpe_mm(pred_cw, gt).item())

    refs = {
        "unweighted_direct_mm": float(np.mean(per_clip_uw)) if per_clip_uw else float("nan"),
        "conf_direct_mm": float(np.mean(per_clip_cw)) if per_clip_cw else float("nan"),
    }
    return refs


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log("=== Iskakov Learnable Triangulation – AIST++ full train/val ===")
    log(f"device={device} torch={torch.__version__} seed={args.seed}")

    split_path = Path(args.split_path)
    train_clips, val_clips = load_split(split_path)
    total_train_frames = sum(c.T for c in train_clips)
    total_val_frames = sum(c.T for c in val_clips)
    log(f"train clips={len(train_clips)} frames={total_train_frames}")
    log(f"val   clips={len(val_clips)} frames={total_val_frames}")

    model = IskakovLearnableTriangulation(hidden_dim=args.hidden_dim, cross_view=True).to(device)
    if device.type == "cpu":
        # CPU oneMKL's float32 lstsq is unstable for AIST++ (NaN detections, near-singular
        # systems).  Use double precision on CPU; GPU runs stay in float32.
        model = model.double()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: IskakovLearnableTriangulation hidden_dim={args.hidden_dim} params={n_params} dtype={next(model.parameters()).dtype}")

    log("computing frozen DLT references on val (batched, all frames)...")
    refs = frozen_dlt_references(model, val_clips, device)
    log(f"FROZEN unweighted_direct_mm = {refs['unweighted_direct_mm']:.2f} mm")
    log(f"FROZEN conf_direct_mm      = {refs['conf_direct_mm']:.2f} mm")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches = max(1, args.train_samples_per_epoch // args.batch_size)
    total_steps = args.epochs * n_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    ckpt_path = Path(args.ckpt_path or log_path.with_suffix(".pth"))
    best = {"combined_direct_mm": float("inf"), "epoch": -1, "metrics": None}
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_root = 0.0
        rng = np.random.default_rng(args.seed + epoch)
        for _ in range(n_batches):
            clip = train_clips[int(rng.integers(len(train_clips)))]
            # Keep drawing clips until we find one with at least one valid frame.
            while clip.valid.sum() == 0:
                clip = train_clips[int(rng.integers(len(train_clips)))]
            idx = rng.integers(0, clip.T, size=args.batch_size)
            batch = clip.sample_frames(idx, device, model)
            if batch is None:
                continue
            p2d, conf, gt, K, R, t = batch
            pred = model(p2d, conf, K, R, t)
            loss = mpjpe_mm(pred, gt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/inf loss at epoch {epoch}")
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            epoch_loss += float(loss)
            epoch_root += float(centroid_aligned_mpjpe_mm(pred, gt))

        # Validation
        model.eval()
        val_metrics: Dict[str, float] = {}
        direct_vals, root_vals = [], []
        for clip in val_clips:
            m = evaluate_clip(model, clip, device)
            if math.isfinite(m["direct_mm"]):
                direct_vals.append(m["direct_mm"])
                root_vals.append(m["root_mm"])
        val_metrics["combined_direct_mm"] = float(np.mean(direct_vals)) if direct_vals else float("nan")
        val_metrics["combined_root_mm"] = float(np.mean(root_vals)) if root_vals else float("nan")
        val_metrics["combined_std_direct_mm"] = float(np.std(direct_vals)) if direct_vals else float("nan")
        val_metrics["combined_std_root_mm"] = float(np.std(root_vals)) if root_vals else float("nan")

        train_mm = epoch_loss / n_batches
        train_root_mm = epoch_root / n_batches
        lr_now = optim.param_groups[0]["lr"]
        log(f"epoch {epoch:03d}/{args.epochs} lr={lr_now:.2e} "
            f"train_MPJPE={train_mm:.2f} mm train_root={train_root_mm:.2f} mm "
            f"val direct={val_metrics['combined_direct_mm']:.2f} mm "
            f"val root={val_metrics['combined_root_mm']:.2f} mm")
        history.append({"epoch": epoch, "train_direct_mm": train_mm, "train_root_mm": train_root_mm,
                        "val_direct_mm": val_metrics["combined_direct_mm"],
                        "val_root_mm": val_metrics["combined_root_mm"]})

        if not all(math.isfinite(m) for m in val_metrics.values()):
            raise RuntimeError(f"NaN/inf val metric at epoch {epoch}")

        if val_metrics["combined_direct_mm"] < best["combined_direct_mm"] - 1e-9:
            best = {"combined_direct_mm": val_metrics["combined_direct_mm"],
                    "epoch": epoch,
                    "metrics": dict(val_metrics)}
            torch.save({"model_state": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)
            epochs_no_improve = 0
            log(f"  new best val direct MPJPE = {val_metrics['combined_direct_mm']:.2f} mm (ckpt {ckpt_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log(f"early stop at epoch {epoch} (patience {args.patience})")
                break

    # Final evaluation from best checkpoint
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        model.eval()
    final_direct, final_root = [], []
    for clip in val_clips:
        m = evaluate_clip(model, clip, device)
        if math.isfinite(m["direct_mm"]):
            final_direct.append(m["direct_mm"])
            final_root.append(m["root_mm"])
    final_combined_direct = float(np.mean(final_direct)) if final_direct else float("nan")
    final_combined_root = float(np.mean(final_root)) if final_root else float("nan")

    log("=== SUMMARY ===")
    log(f"best epoch = {best['epoch']}")
    log(f"FINAL val combined_direct_mm = {final_combined_direct:.2f} mm")
    log(f"FINAL val combined_root_mm   = {final_combined_root:.2f} mm")
    log(f"FROZEN unweighted_direct_mm  = {refs['unweighted_direct_mm']:.2f} mm")
    log(f"FROZEN conf_direct_mm        = {refs['conf_direct_mm']:.2f} mm")
    gain_uw = refs["unweighted_direct_mm"] - final_combined_direct
    gain_cw = refs["conf_direct_mm"] - final_combined_direct
    log(f"learned-vs-unweightedDLT gain (direct) = {gain_uw:+.2f} mm")
    log(f"learned-vs-confDLT      gain (direct) = {gain_cw:+.2f} mm")
    log(f"checkpoint: {ckpt_path} | log: {log_path}")

    cfg_path = log_path.with_suffix(".config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "n_params": n_params,
            "best": best,
            "final_val": {
                "combined_direct_mm": final_combined_direct,
                "combined_root_mm": final_combined_root,
            },
            "frozen_references": refs,
            "history": history,
        }, f, indent=2)
    log(f"config/history: {cfg_path}")
    log_f.close()


if __name__ == "__main__":
    main()
