#!/usr/bin/env python3
"""Launcher for the Iskakov ICCV 2019 learnable-triangulation baseline.

Reads a YAML config (see ``configs/iskakov_h36m_true_gt_v2.yaml``), enforces
the project GPU policy, and runs the trainer in
``experiments/train_iskakov_baseline_shelf_campus.py``. When ``eval.enabled`` is
true and a best checkpoint is produced, the script also runs the sparse-view
MPJPE@k evaluator.

Examples
--------
A800 (GPU 6, default):
    python scripts/run_iskakov_baseline.py --config configs/iskakov_h36m_true_gt_v2.yaml

Local smoke on the RTX 4090:
    python scripts/run_iskakov_baseline.py --config configs/iskakov_h36m_true_gt_v2.yaml \
        --gpu 0 --epochs 1 --train_samples_per_epoch 128 --batch_size 8

Dry-run (print the command without executing):
    python scripts/run_iskakov_baseline.py --config configs/iskakov_h36m_true_gt_v2.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINER = REPO_ROOT / "experiments" / "train_iskakov_baseline_shelf_campus.py"
EVALUATOR = REPO_ROOT / "experiments" / "eval_iskakov_mpjpe_at_k.py"

# Only GPU 6 and GPU 7 may be used on A800.
A800_ALLOWED_GPUS = {"6", "7"}


def _is_a800() -> bool:
    """Heuristic: we are on the A800 host if the repo is on the A800 nvme mount."""
    return "/mnt/nvme0n1p1/zhangzy/" in str(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML config file (e.g. configs/iskakov_h36m_true_gt_v2.yaml)")
    p.add_argument("--gpu", default="6", help="CUDA device to use (default 6)")
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    # Allow any trainer CLI arg to override the config.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--train_samples_per_epoch", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse as a dict")
    return cfg


def flatten_config(cfg: dict) -> dict:
    """Return a flat dict of trainer args from the config file.

    The 'eval' block is kept aside; everything else is passed to the trainer.
    """
    flat = dict(cfg)
    flat.pop("eval", None)
    return flat


def build_trainer_command(cfg: dict, overrides: dict) -> list[str]:
    cmd = [sys.executable, "-u", str(TRAINER)]
    args: list[str] = []

    for key, value in cfg.items():
        if key in ("name",):
            continue
        # Override from CLI if provided.
        if key in overrides and overrides[key] is not None:
            value = overrides[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    return cmd + args


def build_eval_command(checkpoint: Path, cfg: dict, protocol: str) -> list[str]:
    eval_cfg = cfg.get("eval", {})
    cmd = [sys.executable, "-u", str(EVALUATOR)]
    args = [
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--output_dir", str(eval_cfg.get("output_dir", "outputs/iskakov_mpjpe_at_k")),
        "--num_subsets", str(eval_cfg.get("num_subsets", 50)),
        "--max_frames", str(eval_cfg.get("max_frames", 4000)),
    ]
    return cmd + args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    gpu = args.gpu
    if _is_a800() and gpu not in A800_ALLOWED_GPUS:
        raise SystemExit(
            f"A800 GPU policy violation: requested GPU {gpu}. "
            f"Only {A800_ALLOWED_GPUS} may be used by this project."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    # If running on A800, also print a friendly reminder not to disturb v85.
    if _is_a800():
        print(f"[run_iskakov_baseline] A800 detected; using CUDA_VISIBLE_DEVICES={gpu}")
    else:
        print(f"[run_iskakov_baseline] Local/WSL detected; using CUDA_VISIBLE_DEVICES={gpu}")

    flat_cfg = flatten_config(cfg)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "train_samples_per_epoch": args.train_samples_per_epoch,
        "device": args.device,
    }
    trainer_cmd = build_trainer_command(flat_cfg, overrides)

    print("Trainer command:")
    print(" ".join(trainer_cmd))

    if args.dry_run:
        print("\nDry-run: exiting without launching training.")
        return

    # Run training.
    proc = subprocess.run(trainer_cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Training failed with exit code {proc.returncode}")

    # Optionally run MPJPE@k eval if enabled.
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("enabled", False):
        ckpt_path = Path(overrides.get("ckpt_path") or cfg.get("ckpt_path"))
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found, skipping eval: {ckpt_path}", file=sys.stderr)
            return
        eval_cmd = build_eval_command(ckpt_path, cfg, cfg.get("protocol", "h36m_true_gt_v2"))
        print("\nEvaluation command:")
        print(" ".join(eval_cmd))
        proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            raise SystemExit(f"Evaluation failed with exit code {proc.returncode}")


if __name__ == "__main__":
    main()
