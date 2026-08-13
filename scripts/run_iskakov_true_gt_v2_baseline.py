#!/usr/bin/env python3
"""Reproduce/evaluate the Iskakov et al. ICCV 2019 learnable-triangulation
baseline on the H36M true-GT v2 protocol (S1,5,6,7,8 -> S9/S11).

This script is a thin wrapper around
``experiments/train_iskakov_baseline_shelf_campus.py`` with the canonical
hyperparameters that produced the project's reported 23.40 mm MPJPE on
h36m_true_gt_v2.  It also optionally runs the sparse-view MPJPE@k evaluator.

Usage
-----
Local / WSL (GPU 0):
    python scripts/run_iskakov_true_gt_v2_baseline.py --gpu 0 --epochs 1 \
        --train_samples_per_epoch 128

A800 (GPU 6 or 7, default GPU 6):
    python scripts/run_iskakov_true_gt_v2_baseline.py

Dry-run:
    python scripts/run_iskakov_true_gt_v2_baseline.py --dry-run

References
----------
* Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y., "Learnable Triangulation
  of Human Pose", ICCV 2019, arXiv:1905.05754.
* Project result: H36M true-GT v2 val MPJPE 23.40 mm.

Assumptions and limitations
---------------------------
* This script reproduces the *learnable triangulation* (weight-prediction)
  branch of Iskakov et al., not the volumetric/triangulation network branch.
* The 23.40 mm target is obtained with the exact hyperparameters below and the
  H36M true-GT v2 split.  Using ``h36m_true_gt`` (v1) or the old circular
  ``data/h36m_hf`` split will give a different, non-comparable number.
* Inputs are the canonical ``.npz`` files in ``data/h36m_true_gt_v2/``, which
  already contain GT-projected 2D keypoints and per-joint confidences.  This
  baseline does **not** run an external 2D detector such as RTMPose; it only
  triangulates the stored 2D.
* Triangulation is performed by the differentiable weighted-LS DLT solver in
  ``motionflow_mv.fusion.triangulation.triangulate_dlt_batched_lstsq``.  The
  learned component is a small MLP that predicts per-view per-joint weights.
* Sparse-view (k<4) evaluation is a post-hoc diagnostic (MPJPE@k); the model is
  trained on all 4 views and may fail catastrophically for k<4 unless explicitly
  trained with view dropout.
* GPU policy: on A800 only CUDA devices 6 and 7 are allowed; the wrapper exits
  if another GPU is requested.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINER = REPO_ROOT / "experiments" / "train_iskakov_baseline_shelf_campus.py"
EVALUATOR = REPO_ROOT / "experiments" / "eval_iskakov_mpjpe_at_k.py"

# MotionFlow-MultiView only ever uses GPUs 6 and 7 on A800.
A800_ALLOWED_GPUS = {"6", "7"}


def _is_a800() -> bool:
    """Heuristic: repo lives on the A800 nvme mount."""
    return "/mnt/nvme0n1p1/zhangzy/" in str(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="6", help="CUDA device to use (default 6)")
    p.add_argument("--dry-run", action="store_true", help="print command and exit")
    # Hyperparameters that produced the reported 23.40 mm result.
    p.add_argument("--epochs", type=int, default=10, help="training epochs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--train_samples_per_epoch", type=int, default=4096)
    p.add_argument("--ref_max_frames", type=int, default=2000)
    p.add_argument("--eval_mpjpe_at_k", action="store_true", default=True,
                   help="run MPJPE@k sparse-view eval after training")
    p.add_argument("--num_subsets", type=int, default=50)
    p.add_argument("--max_frames", type=int, default=4000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_path", type=str, default=None)
    p.add_argument("--ckpt_path", type=str, default=None)
    return p.parse_args()


def build_trainer_command(args: argparse.Namespace) -> list[str]:
    log_path = Path(args.log_path or "outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.log")
    ckpt_path = Path(args.ckpt_path or "outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.pth")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(TRAINER),
        "--protocol", "h36m_true_gt_v2",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--grad_clip", str(args.grad_clip),
        "--seed", str(args.seed),
        "--hidden_dim", str(args.hidden_dim),
        "--train_samples_per_epoch", str(args.train_samples_per_epoch),
        "--ref_max_frames", str(args.ref_max_frames),
        "--log_path", str(log_path),
        "--ckpt_path", str(ckpt_path),
        "--device", args.device,
    ]
    return cmd


def build_eval_command(args: argparse.Namespace, ckpt_path: Path) -> list[str]:
    output_dir = Path("outputs/iskakov_mpjpe_at_k_h36m_true_gt_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, "-u", str(EVALUATOR),
        "--protocol", "h36m_true_gt_v2",
        "--checkpoint", str(ckpt_path),
        "--hidden_dim", str(args.hidden_dim),
        "--output_dir", str(output_dir),
        "--num_subsets", str(args.num_subsets),
        "--max_frames", str(args.max_frames),
        "--device", args.device,
    ]


def main() -> None:
    args = parse_args()

    gpu = args.gpu
    if _is_a800() and gpu not in A800_ALLOWED_GPUS:
        raise SystemExit(
            f"A800 GPU policy violation: requested GPU {gpu}. "
            f"Only {A800_ALLOWED_GPUS} may be used by this project."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    trainer_cmd = build_trainer_command(args)
    ckpt_path = Path(args.ckpt_path or "outputs/baselines/iskakov_learnable_tri_h36m_true_gt_v2.pth")

    print("[run_iskakov_true_gt_v2_baseline] Iskakov ICCV 2019 H36M true-GT v2 baseline")
    if _is_a800():
        print(f"  A800 detected; CUDA_VISIBLE_DEVICES={gpu}")
    else:
        print(f"  Local/WSL detected; CUDA_VISIBLE_DEVICES={gpu}")
    print(f"  checkpoint will be saved to: {ckpt_path}")
    print("Trainer command:")
    print(" ".join(trainer_cmd))

    if args.dry_run:
        print("\nDry-run: exiting without launching training.")
        return

    proc = subprocess.run(trainer_cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Training failed with exit code {proc.returncode}")

    if args.eval_mpjpe_at_k:
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found, skipping MPJPE@k eval: {ckpt_path}", file=sys.stderr)
            return
        eval_cmd = build_eval_command(args, ckpt_path)
        print("\nMPJPE@k evaluation command:")
        print(" ".join(eval_cmd))
        proc = subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            raise SystemExit(f"MPJPE@k evaluation failed with exit code {proc.returncode}")


if __name__ == "__main__":
    main()
