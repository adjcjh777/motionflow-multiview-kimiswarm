"""Large-run hyperparameter search for MPI-INF-3DHP.

Searches over the most impactful architecture and augmentation knobs for the
Bayesian triangulation v2 pipeline.  Each trial calls the existing trainer
``experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py``
so the search stays in sync with the current codebase.

Usage
-----
CPU smoke test (two tiny trials, 1 epoch each) on the smoke .npz files::

    python experiments/prototypes/hp_search_large_mpiinf3dhp.py \
        --mode smoke --n_trials 2 --epochs 1 --device cpu

Small random search on a single GPU (queued behind the anchor run) using the
smoke files so it finishes quickly::

    python experiments/prototypes/hp_search_large_mpiinf3dhp.py \
        --mode random --n_trials 8 --epochs 5 --device cuda

Full large-run search (intended to run after the anchor 50-epoch run) on the
full MPI-INF-3DHP WebBridge data::

    python experiments/prototypes/hp_search_large_mpiinf3dhp.py \
        --mode random --n_trials 20 --epochs 50 --device cuda --full

Results are written to ``outputs/hp_search_large_mpiinf3dhp/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


ROOT = Path(__file__).parent.parent.parent
TRAINER = ROOT / "experiments" / "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "hp_search_large_mpiinf3dhp"


@dataclass
class SearchSpace:
    """Hyperparameter search space for large MPI-INF-3DHP runs."""

    d: List[int]
    residual_hidden: List[int]
    n_st_layers: List[int]
    lr: List[float]
    pp_loss_weight: List[float]
    epipolar_loss_weight: List[float]
    cam_aug_pp: List[float]
    cam_aug_focal: List[float]
    batch_size: List[int]
    train_samples: List[int]

    def sample_random(self, rng: random.Random) -> Dict[str, Any]:
        return {
            "d": rng.choice(self.d),
            "residual_hidden": rng.choice(self.residual_hidden),
            "n_st_layers": rng.choice(self.n_st_layers),
            "lr": rng.choice(self.lr),
            "pp_loss_weight": rng.choice(self.pp_loss_weight),
            "epipolar_loss_weight": rng.choice(self.epipolar_loss_weight),
            "cam_aug_pp": rng.choice(self.cam_aug_pp),
            "cam_aug_focal": rng.choice(self.cam_aug_focal),
            "batch_size": rng.choice(self.batch_size),
            "train_samples": rng.choice(self.train_samples),
        }


@dataclass
class TrialConfig:
    """One concrete hyperparameter configuration."""

    trial_id: int
    d: int
    residual_hidden: int
    n_st_layers: int
    lr: float
    pp_loss_weight: float
    epipolar_loss_weight: float
    cam_aug_pp: float
    cam_aug_focal: float
    batch_size: int
    train_samples: int

    def slug(self) -> str:
        return (
            f"t{self.trial_id:03d}_d{self.d}_rh{self.residual_hidden}_"
            f"nst{self.n_st_layers}_lr{self.lr:.0e}_pp{self.pp_loss_weight}_"
            f"epi{self.epipolar_loss_weight}_bpp{self.cam_aug_pp}_"
            f"focal{self.cam_aug_focal}_bs{self.batch_size}_ts{self.train_samples}"
        )


def default_search_space() -> SearchSpace:
    return SearchSpace(
        d=[64, 128, 256],
        residual_hidden=[128, 256, 512],
        n_st_layers=[2, 3, 4],
        lr=[3e-4, 1e-3, 3e-3],
        pp_loss_weight=[0.1, 0.2, 0.5],
        epipolar_loss_weight=[0.0, 0.05, 0.1],
        cam_aug_pp=[3.0, 5.0, 8.0],
        cam_aug_focal=[0.005, 0.01, 0.02],
        batch_size=[4, 8, 16],
        train_samples=[1000, 2000, 4000],
    )


def smoke_search_space() -> SearchSpace:
    """Narrow space used for the CPU smoke test."""
    return SearchSpace(
        d=[32, 64],
        residual_hidden=[64, 128],
        n_st_layers=[1, 2],
        lr=[1e-3, 3e-4],
        pp_loss_weight=[0.1, 0.2],
        epipolar_loss_weight=[0.0, 0.05],
        cam_aug_pp=[2.0, 5.0],
        cam_aug_focal=[0.005, 0.01],
        batch_size=[2, 4],
        train_samples=[100, 200],
    )


def generate_trials(
    space: SearchSpace,
    mode: str,
    n_trials: int,
    seed: int,
) -> List[TrialConfig]:
    if mode == "grid":
        keys = [
            "d",
            "residual_hidden",
            "n_st_layers",
            "lr",
            "pp_loss_weight",
            "epipolar_loss_weight",
            "cam_aug_pp",
            "cam_aug_focal",
            "batch_size",
            "train_samples",
        ]
        values = [getattr(space, k) for k in keys]
        combos = list(itertools.product(*values))
        rng = random.Random(seed)
        rng.shuffle(combos)
        trials: List[TrialConfig] = []
        for i, combo in enumerate(combos[:n_trials]):
            kwargs = dict(zip(keys, combo))
            trials.append(TrialConfig(trial_id=i, **kwargs))
        return trials

    # random
    rng = random.Random(seed)
    trials = []
    for i in range(n_trials):
        kwargs = space.sample_random(rng)
        trials.append(TrialConfig(trial_id=i, **kwargs))
    return trials


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Large-run hyperparameter search for MPI-INF-3DHP",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "grid", "smoke"],
        help="Search mode: random (default), grid, or smoke (tiny random CPU run)",
    )
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials (random) or max grid combinations")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per trial")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for training")
    parser.add_argument("--full", action="store_true", help="Use full .npz files instead of smoke files")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(ROOT / "data" / "webbridge" / "mpi_inf_3dhp"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing trials.json to resume from",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be run, but do not execute them",
    )
    args = parser.parse_args(argv)
    return args


def _data_paths(args: argparse.Namespace) -> Tuple[List[Path], Path]:
    """Return (train_paths, val_path) depending on smoke vs. full mode."""
    suffix = "" if args.full else "_smoke"
    if args.full:
        train_names = [
            "s_01_seq_01_v14_multiview_m.npz",
            "s_01_seq_02_v14_multiview_m.npz",
            "s_03_seq_01_v14_multiview_m.npz",
            "s_03_seq_02_v14_multiview_m.npz",
        ]
    else:
        # Smoke files only exist for s_01 and s_02.
        train_names = [
            "s_01_seq_01_v14_multiview_m_smoke.npz",
            "s_01_seq_02_v14_multiview_m_smoke.npz",
        ]
    train_paths = [Path(args.data_root) / name for name in train_names]
    val_path = Path(args.data_root) / f"s_02_seq_01_v14_multiview_m{suffix}.npz"
    return train_paths, val_path


def build_command(
    trial: TrialConfig,
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> List[str]:
    train_paths, val_path = _data_paths(args)

    if not dry_run:
        for p in train_paths + [val_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {p}")

    output_path = Path(args.output_dir) / f"{trial.slug()}.pth"

    cmd = [
        sys.executable,
        str(TRAINER),
        "--train",
        *[str(p) for p in train_paths],
        "--val",
        str(val_path),
        "--clip_len", "13",
        "--d", str(trial.d),
        "--residual_hidden", str(trial.residual_hidden),
        "--n_st_layers", str(trial.n_st_layers),
        "--model_type", "bayesian_tri_v2",
        "--epochs", str(args.epochs),
        "--train_samples", str(trial.train_samples),
        "--batch_size", str(trial.batch_size),
        "--val_stride", "50",
        "--lr", str(trial.lr),
        "--pp_loss_weight", str(trial.pp_loss_weight),
        "--epipolar_loss_weight", str(trial.epipolar_loss_weight),
        "--cam_aug_pp", str(trial.cam_aug_pp),
        "--cam_aug_focal", str(trial.cam_aug_focal),
        "--cam_aug_schedule", "intrinsics_curriculum",
        "--cam_aug_intrinsics_ramp_epochs", "5",
        "--pp_pretrain_epochs", "3",
        "--output", str(output_path),
    ]
    return cmd


def run_trial(
    cmd: List[str],
    dry_run: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training trial and return its metrics."""
    result: Dict[str, Any] = {
        "command": " ".join(cmd),
        "returncode": None,
        "stdout_tail": "",
        "best_val_mpjpe_mm": None,
        "elapsed_sec": None,
    }
    if dry_run:
        result["returncode"] = 0
        return result

    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        lines.append(line)
    proc.wait()
    elapsed = time.time() - start

    result["returncode"] = proc.returncode
    result["elapsed_sec"] = elapsed
    result["stdout_tail"] = "".join(lines[-200:])

    # Parse best val MPJPE from the trainer's stdout.
    best_val = None
    for line in lines:
        if "Best val MPJPE:" in line:
            try:
                token = line.split("Best val MPJPE:")[1].split("mm")[0].strip()
                best_val = float(token)
            except (ValueError, IndexError):
                pass
    result["best_val_mpjpe_mm"] = best_val
    return result


def write_report(trials: List[Dict[str, Any]], output_dir: Path) -> None:
    report_path = output_dir / "hp_search_report.md"
    with open(report_path, "w") as f:
        f.write("# Hyperparameter Search Report: Large MPI-INF-3DHP Runs\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total trials: {len(trials)}\n")
        f.write(f"- Successful trials: {sum(1 for t in trials if t.get('returncode') == 0)}\n")
        f.write(f"- Failed trials: {sum(1 for t in trials if t.get('returncode') != 0)}\n\n")

        completed = [t for t in trials if t.get("best_val_mpjpe_mm") is not None]
        if completed:
            best = min(completed, key=lambda t: t["best_val_mpjpe_mm"])
            f.write("## Best trial\n\n")
            f.write(f"- Trial ID: {best['trial_id']}\n")
            f.write(f"- Best val MPJPE: {best['best_val_mpjpe_mm']:.2f} mm\n")
            f.write(f"- Slug: `{best['slug']}`\n")
            f.write(f"- Checkpoint: `{best['output']}`\n\n")

        f.write("## All trials\n\n")
        f.write(
            "| ID | d | residual_hidden | n_st_layers | lr | pp_loss | epi_loss | cam_aug_pp | "
            "cam_aug_focal | bs | train_samples | returncode | best_val_mm | elapsed_min |\n"
        )
        f.write(
            "|----|---|-----------------|-------------|----|---------|----------|------------|"
            "--------------|----|---------------|------------|-------------|-------------|\n"
        )
        for t in trials:
            cfg = t["config"]
            best = t.get("best_val_mpjpe_mm")
            best_str = f"{best:.2f}" if best is not None else "n/a"
            elapsed_min = "n/a" if t.get("elapsed_sec") is None else f"{t['elapsed_sec']/60:.1f}"
            f.write(
                f"| {t['trial_id']} | {cfg['d']} | {cfg['residual_hidden']} | {cfg['n_st_layers']} | "
                f"{cfg['lr']:.0e} | {cfg['pp_loss_weight']} | {cfg['epipolar_loss_weight']} | "
                f"{cfg['cam_aug_pp']} | {cfg['cam_aug_focal']} | {cfg['batch_size']} | "
                f"{cfg['train_samples']} | {t.get('returncode', 'n/a')} | {best_str} | {elapsed_min} |\n"
            )
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.mode == "smoke":
        space = smoke_search_space()
        n_trials = args.n_trials
        epochs = min(args.epochs, 2)
        device = "cpu"
    else:
        space = default_search_space()
        n_trials = args.n_trials
        epochs = args.epochs
        device = args.device

    # Smoke mode forces CPU and uses the smoke .npz files regardless of --full.
    args.full = False if args.mode == "smoke" else args.full

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trials = generate_trials(space, args.mode, n_trials, args.seed)

    # Load existing results if resuming.
    trials_json_path = output_dir / "trials.json"
    if args.resume:
        trials_json_path = Path(args.resume)
    results: List[Dict[str, Any]] = []
    completed_ids: set = set()
    if trials_json_path.exists():
        with open(trials_json_path) as f:
            results = json.load(f)
        completed_ids = {r["trial_id"] for r in results}
        print(f"Resumed from {trials_json_path} ({len(results)} previous trials)")

    try:
        for trial in trials:
            if trial.trial_id in completed_ids:
                print(f"Skipping already-completed trial {trial.trial_id}")
                continue

            print(f"\n=== Trial {trial.trial_id:03d}/{len(trials):03d}: {trial.slug()} ===")
            cmd = build_command(trial, args, dry_run=args.dry_run)

            trial_record: Dict[str, Any] = {
                "trial_id": trial.trial_id,
                "slug": trial.slug(),
                "config": asdict(trial),
                "output": str(Path(args.output_dir) / f"{trial.slug()}.pth"),
            }

            result = run_trial(cmd, dry_run=args.dry_run, device=device)
            trial_record.update(result)
            results.append(trial_record)

            # Save after every trial so crashes can be resumed.
            with open(trials_json_path, "w") as f:
                json.dump(results, f, indent=2)

            if result["returncode"] != 0:
                print(f"  Trial {trial.trial_id} failed with return code {result['returncode']}")
            else:
                print(
                    f"  Trial {trial.trial_id} done: best_val_mpjpe_mm="
                    f"{result.get('best_val_mpjpe_mm')}, elapsed={(result.get('elapsed_sec') or 0)/60:.1f} min"
                )
    finally:
        write_report(results, output_dir)
        print(f"\nReport written to {output_dir / 'hp_search_report.md'}")
        print(f"Trials JSON: {trials_json_path}")


if __name__ == "__main__":
    main()
