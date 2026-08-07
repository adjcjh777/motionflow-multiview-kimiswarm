"""Swarm-iter18 large-run hyperparameter search (v2).

This script drives long-running MPI-INF-3DHP experiments for the
``feat/swarm-iter18-omniview`` branch.  It is trainer-agnostic: by default it
calls the current best trainer
(``train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py``),
but it can also target the upcoming ``train_omnimultiview_mpiinf3dhp.py`` trainer
once that file lands.

Key improvements over ``hp_search_large_mpiinf3dhp.py``:

- **ASHA-style successive halving** prunes bad trials early instead of running
  every trial to the full epoch budget.
- **Warm-start support** from an anchor checkpoint (e.g. the 9.32 mm PP model
  or the 8.35 mm Bayesian tri v2 ensemble member).
- **Trainer-agnostic** ``--trainer`` flag so the same search harness can be used
  for the legacy backbone and for OmniMultiViewFusion.
- **Extended search space** includes visibility gating, graph-joint layers, view
  dropout, and uncertainty/auxiliary-loss weights required by the omniview
  architecture.
- **Smarter resume**: each trial writes its own partial log; the orchestrator
  only re-runs trials whose checkpoints are missing.

Usage
-----
CPU smoke / dry-run (no GPU, no real trainer invocation) to validate the
search harness itself::

    python experiments/prototypes/swarm_iter18/hyperparameter_search_v2.py \
        --mode smoke --n_trials 3 --epochs 1 --device cpu --dry_run

Small random search on the smoke .npz files (fast GPU run) behind the anchor
run::

    python experiments/prototypes/swarm_iter18/hyperparameter_search_v2.py \
        --mode random --n_trials 8 --epochs 10 --device cuda

Full large-run search on the full WebBridge MPI-INF-3DHP data with ASHA
successive halving (20 trials, 50 epochs max, 4 rungs) and warm-start::

    python experiments/prototypes/swarm_iter18/hyperparameter_search_v2.py \
        --mode random --n_trials 20 --epochs 50 --asha_rungs 4 \
        --warm_start outputs/ray_attention_temporal_crossview_residual_principal_point_curriculum_v1.pth \
        --device cuda --full

Results are written to ``outputs/swarm_iter18_hp_search_v2/``.

Use ``--omniview`` to include OmniMultiViewFusion-specific flags in the search
space (e.g. ``--uncertainty_loss_weight`` and ``--graph_attention_heads``).
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_TRAINER = (
    ROOT
    / "experiments"
    / "train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "swarm_iter18_hp_search_v2"


@dataclass
class SearchSpace:
    """Hyperparameter search space for swarm-iter18 large runs.

    The space is intentionally broad enough to cover both the legacy
    RayAttention PP backbone and the future OmniMultiViewFusion trainer.
    Parameters that a trainer does not recognise are simply not passed.
    """

    # Architecture
    d: List[int]
    residual_hidden: List[int]
    n_st_layers: List[int]
    n_view_layers: List[int]
    n_temporal_layers: List[int]
    principal_point_hidden: List[int]

    # Optimisation
    lr: List[float]
    lr_warmup_epochs: List[int]
    lr_cosine: List[bool]
    batch_size: List[int]
    train_samples: List[int]
    grad_clip_norm: List[float]

    # Camera augmentation / intrinsic correction
    pp_loss_weight: List[float]
    focal_loss_weight: List[float]
    cam_aug_pp: List[float]
    cam_aug_focal: List[float]
    cam_aug_rot: List[float]
    cam_aug_trans: List[float]
    cam_aug_schedule: List[str]

    # Auxiliary losses (legacy + omniview)
    epipolar_loss_weight: List[float]
    visibility_loss_weight: List[float]
    uncertainty_loss_weight: List[float]
    bone_loss_weight: List[float]
    velocity_loss_weight: List[float]

    # OmniMultiViewFusion-specific
    view_dropout_rate: List[float]
    min_views: List[int]
    n_joint_graph_layers: List[int]
    graph_num_layers: List[int]
    graph_attention_heads: List[int]

    def sample_random(self, rng: random.Random) -> Dict[str, Any]:
        return {
            "d": rng.choice(self.d),
            "residual_hidden": rng.choice(self.residual_hidden),
            "n_st_layers": rng.choice(self.n_st_layers),
            "n_view_layers": rng.choice(self.n_view_layers),
            "n_temporal_layers": rng.choice(self.n_temporal_layers),
            "principal_point_hidden": rng.choice(self.principal_point_hidden),
            "lr": rng.choice(self.lr),
            "lr_warmup_epochs": rng.choice(self.lr_warmup_epochs),
            "lr_cosine": rng.choice(self.lr_cosine),
            "batch_size": rng.choice(self.batch_size),
            "train_samples": rng.choice(self.train_samples),
            "grad_clip_norm": rng.choice(self.grad_clip_norm),
            "pp_loss_weight": rng.choice(self.pp_loss_weight),
            "focal_loss_weight": rng.choice(self.focal_loss_weight),
            "cam_aug_pp": rng.choice(self.cam_aug_pp),
            "cam_aug_focal": rng.choice(self.cam_aug_focal),
            "cam_aug_rot": rng.choice(self.cam_aug_rot),
            "cam_aug_trans": rng.choice(self.cam_aug_trans),
            "cam_aug_schedule": rng.choice(self.cam_aug_schedule),
            "epipolar_loss_weight": rng.choice(self.epipolar_loss_weight),
            "visibility_loss_weight": rng.choice(self.visibility_loss_weight),
            "uncertainty_loss_weight": rng.choice(self.uncertainty_loss_weight),
            "bone_loss_weight": rng.choice(self.bone_loss_weight),
            "velocity_loss_weight": rng.choice(self.velocity_loss_weight),
            "view_dropout_rate": rng.choice(self.view_dropout_rate),
            "min_views": rng.choice(self.min_views),
            "n_joint_graph_layers": rng.choice(self.n_joint_graph_layers),
            "graph_num_layers": rng.choice(self.graph_num_layers),
            "graph_attention_heads": rng.choice(self.graph_attention_heads),
        }


@dataclass
class TrialConfig:
    """One concrete hyperparameter configuration."""

    trial_id: int
    d: int
    residual_hidden: int
    n_st_layers: int
    n_view_layers: int
    n_temporal_layers: int
    principal_point_hidden: int
    lr: float
    lr_warmup_epochs: int
    lr_cosine: bool
    batch_size: int
    train_samples: int
    grad_clip_norm: float
    pp_loss_weight: float
    focal_loss_weight: float
    cam_aug_pp: float
    cam_aug_focal: float
    cam_aug_rot: float
    cam_aug_trans: float
    cam_aug_schedule: str
    epipolar_loss_weight: float
    visibility_loss_weight: float
    uncertainty_loss_weight: float
    bone_loss_weight: float
    velocity_loss_weight: float
    view_dropout_rate: float
    min_views: int
    n_joint_graph_layers: int
    graph_num_layers: int
    graph_attention_heads: int

    # Derived rung state (not part of the slug)
    rung: int = field(default=0, compare=False)
    target_epochs: int = field(default=0, compare=False)

    def slug(self) -> str:
        return (
            f"t{self.trial_id:03d}_d{self.d}_rh{self.residual_hidden}_"
            f"nst{self.n_st_layers}_nvw{self.n_view_layers}_ntmp{self.n_temporal_layers}_"
            f"lr{self.lr:.0e}_pp{self.pp_loss_weight}_epi{self.epipolar_loss_weight}_"
            f"vis{self.visibility_loss_weight}_unc{self.uncertainty_loss_weight}_"
            f"vd{self.view_dropout_rate}_mv{self.min_views}_"
            f"bs{self.batch_size}_ts{self.train_samples}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_search_space(omniview: bool = False) -> SearchSpace:
    uncertainty_loss_weight = [0.0, 0.03, 0.05, 0.1] if omniview else [0.0]
    bone_loss_weight = [0.0, 0.02, 0.05] if omniview else [0.0]
    graph_attention_heads = [1, 2, 4] if omniview else [1]
    return SearchSpace(
        d=[64, 128, 256],
        residual_hidden=[128, 256, 512],
        n_st_layers=[2, 3, 4],
        n_view_layers=[2, 3],
        n_temporal_layers=[2, 3],
        principal_point_hidden=[32, 64, 128],
        lr=[3e-4, 1e-3, 3e-3],
        lr_warmup_epochs=[0, 3, 5],
        lr_cosine=[False, True],
        batch_size=[4, 8, 16],
        train_samples=[1000, 2000, 4000],
        grad_clip_norm=[0.0, 1.0, 5.0],
        pp_loss_weight=[0.0, 0.1, 0.2, 0.5],
        focal_loss_weight=[0.0, 0.05, 0.1],
        cam_aug_pp=[2.0, 5.0, 8.0],
        cam_aug_focal=[0.005, 0.01, 0.02],
        cam_aug_rot=[0.3, 0.5, 0.7],
        cam_aug_trans=[0.003, 0.005, 0.007],
        cam_aug_schedule=["flat", "intrinsics_curriculum", "extended_curriculum"],
        epipolar_loss_weight=[0.0, 0.05, 0.1],
        visibility_loss_weight=[0.0, 0.05, 0.1, 0.2],
        uncertainty_loss_weight=uncertainty_loss_weight,
        bone_loss_weight=bone_loss_weight,
        velocity_loss_weight=[0.0, 0.01, 0.02],
        view_dropout_rate=[0.0, 0.1, 0.2, 0.3],
        min_views=[2, 3],
        n_joint_graph_layers=[1, 2],
        graph_num_layers=[1, 2],
        graph_attention_heads=graph_attention_heads,
    )


def smoke_search_space(omniview: bool = False) -> SearchSpace:
    """Narrow space used for CPU/dry-run smoke tests."""
    uncertainty_loss_weight = [0.0, 0.05] if omniview else [0.0]
    bone_loss_weight = [0.0, 0.02] if omniview else [0.0]
    graph_attention_heads = [1, 2] if omniview else [1]
    return SearchSpace(
        d=[32, 64],
        residual_hidden=[64, 128],
        n_st_layers=[1, 2],
        n_view_layers=[1, 2],
        n_temporal_layers=[1, 2],
        principal_point_hidden=[32, 64],
        lr=[1e-3, 3e-4],
        lr_warmup_epochs=[0, 1],
        lr_cosine=[False],
        batch_size=[2, 4],
        train_samples=[100, 200],
        grad_clip_norm=[0.0, 1.0],
        pp_loss_weight=[0.0, 0.1],
        focal_loss_weight=[0.0, 0.05],
        cam_aug_pp=[2.0, 5.0],
        cam_aug_focal=[0.005, 0.01],
        cam_aug_rot=[0.3, 0.5],
        cam_aug_trans=[0.003, 0.005],
        cam_aug_schedule=["flat"],
        epipolar_loss_weight=[0.0, 0.05],
        visibility_loss_weight=[0.0, 0.1],
        uncertainty_loss_weight=uncertainty_loss_weight,
        bone_loss_weight=bone_loss_weight,
        velocity_loss_weight=[0.0, 0.01],
        view_dropout_rate=[0.0, 0.1],
        min_views=[2],
        n_joint_graph_layers=[1],
        graph_num_layers=[1],
        graph_attention_heads=graph_attention_heads,
    )


def generate_trials(
    space: SearchSpace,
    mode: str,
    n_trials: int,
    seed: int,
) -> List[TrialConfig]:
    """Generate ``n_trials`` configurations from ``space``.

    ``mode`` is one of ``random``, ``grid``, or ``smoke``.  For ``grid`` the
    function samples the first ``n_trials`` shuffled combinations.
    """
    if mode == "grid":
        keys = list(space.__annotations__.keys())
        values = [getattr(space, k) for k in keys]
        combos = list(itertools.product(*values))
        rng = random.Random(seed)
        rng.shuffle(combos)
        trials: List[TrialConfig] = []
        for i, combo in enumerate(combos[:n_trials]):
            kwargs = dict(zip(keys, combo))
            trials.append(TrialConfig(trial_id=i, **kwargs))
        return trials

    rng = random.Random(seed)
    trials = []
    for i in range(n_trials):
        kwargs = space.sample_random(rng)
        trials.append(TrialConfig(trial_id=i, **kwargs))
    return trials


def asha_rung_epochs(max_epochs: int, rungs: int) -> List[int]:
    """Return rung epoch budgets for successive halving.

    Example: max_epochs=50, rungs=4 -> [6, 12, 25, 50].
    """
    if rungs < 1:
        raise ValueError("rungs must be >= 1")
    if rungs == 1:
        return [max_epochs]
    # Geometrically spaced rungs, last rung always max_epochs.
    budgets = [max(1, int(max_epochs * ((r + 1) / rungs) ** 2)) for r in range(rungs)]
    budgets[-1] = max_epochs
    return budgets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swarm-iter18 large-run hyperparameter search (v2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "grid", "smoke"],
        help="Search mode: random (default), grid, or smoke (tiny CPU/dry-run)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=16, help="Number of trials (random) or max grid combinations"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs per trial")
    parser.add_argument(
        "--asha_rungs",
        type=int,
        default=1,
        help="Number of ASHA successive-halving rungs (1 = no early pruning)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")
    parser.add_argument(
        "--full", action="store_true", help="Use full .npz files instead of smoke files"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default=str(DEFAULT_TRAINER),
        help="Path to the trainer script to invoke for each trial",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bayesian_tri_v2",
        help="model_type argument passed to the trainer",
    )
    parser.add_argument(
        "--omniview",
        action="store_true",
        help="Use the OmniMultiViewFusion search space (includes uncertainty, bone, graph-head flags)",
    )
    parser.add_argument(
        "--warm_start",
        type=str,
        default=None,
        help="Path to anchor checkpoint to warm-start every trial from",
    )
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
        "--skip_existing",
        action="store_true",
        help="Skip rungs whose checkpoint already exists on disk",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands but do not execute them",
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
        train_names = [
            "s_01_seq_01_v14_multiview_m_smoke.npz",
            "s_01_seq_02_v14_multiview_m_smoke.npz",
        ]
    train_paths = [Path(args.data_root) / name for name in train_names]
    val_path = Path(args.data_root) / f"s_02_seq_01_v14_multiview_m{suffix}.npz"
    return train_paths, val_path


# Map TrialConfig field names to trainer CLI argument names.  Values of None
# mean the parameter is omitted unless non-default (handled below).
def _trial_fields(omniview: bool = False) -> Dict[str, Optional[str]]:
    return {
        "d": "--d",
        "residual_hidden": "--residual_hidden",
        "n_st_layers": "--n_st_layers",
        "n_view_layers": "--n_view_layers",
        "n_temporal_layers": "--n_temporal_layers",
        "principal_point_hidden": "--principal_point_hidden",
        "lr": "--lr",
        "lr_warmup_epochs": "--lr_warmup_epochs",
        "lr_cosine": "--lr_cosine",
        "batch_size": "--batch_size",
        "train_samples": "--train_samples",
        "grad_clip_norm": "--grad_clip_norm",
        "pp_loss_weight": "--pp_loss_weight",
        "focal_loss_weight": "--focal_loss_weight",
        "cam_aug_pp": "--cam_aug_pp",
        "cam_aug_focal": "--cam_aug_focal",
        "cam_aug_rot": "--cam_aug_rot",
        "cam_aug_trans": "--cam_aug_trans",
        "cam_aug_schedule": "--cam_aug_schedule",
        "epipolar_loss_weight": "--epipolar_loss_weight",
        "visibility_loss_weight": "--visibility_loss_weight",
        # OmniMultiViewFusion-only flags.
        "uncertainty_loss_weight": "--uncertainty_loss_weight" if omniview else None,
        "bone_loss_weight": "--bone_loss_weight" if omniview else None,
        "velocity_loss_weight": "--velocity_loss_weight",
        "view_dropout_rate": "--view_dropout_rate",
        "min_views": "--min_views",
        "n_joint_graph_layers": "--n_joint_graph_layers",
        "graph_num_layers": "--graph_num_layers",
        "graph_attention_heads": "--graph_attention_heads" if omniview else None,
    }


def build_command(
    trial: TrialConfig,
    args: argparse.Namespace,
    epochs: int,
    output_path: Path,
    *,
    dry_run: bool = False,
) -> List[str]:
    """Build the trainer command list for a single rung."""
    train_paths, val_path = _data_paths(args)

    if not dry_run:
        for p in train_paths + [val_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {p}")

    cmd: List[str] = [
        sys.executable,
        str(args.trainer),
        "--train",
        *[str(p) for p in train_paths],
        "--val",
        str(val_path),
        "--clip_len",
        "13",
        "--model_type",
        str(args.model_type),
        "--epochs",
        str(epochs),
        "--seed",
        str(args.seed + trial.trial_id),
        "--output",
        str(output_path),
    ]

    if args.warm_start:
        cmd.extend(["--warm_start", str(args.warm_start)])

    # Add trial-specific flags.  We always pass the value so the search space
    # is respected, except for store-true style flags where False means "omit".
    for field_name, flag in _trial_fields(omniview=getattr(args, "omniview", False)).items():
        value = getattr(trial, field_name)
        if flag is None:
            continue
        if field_name == "lr_cosine" and not value:
            continue
        cmd.extend([flag, str(value)])

    return cmd


def run_trial(
    cmd: List[str],
    *,
    dry_run: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training command and return metrics."""
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

    best_val: Optional[float] = None
    for line in lines:
        if "Best val MPJPE:" in line:
            try:
                token = line.split("Best val MPJPE:")[1].split("mm")[0].strip()
                best_val = float(token)
            except (ValueError, IndexError):
                pass
    result["best_val_mpjpe_mm"] = best_val
    return result


def write_report(
    trials: List[Dict[str, Any]],
    output_dir: Path,
    budgets: List[int],
) -> None:
    """Write a markdown summary of the search to ``output_dir``."""
    report_path = output_dir / "hp_search_report.md"
    with open(report_path, "w") as f:
        f.write("# Swarm-iter18 Hyperparameter Search Report (v2)\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total trial records: {len(trials)}\n")
        f.write(f"- Successful records: {sum(1 for t in trials if t.get('returncode') == 0)}\n")
        f.write(f"- Failed records: {sum(1 for t in trials if t.get('returncode') != 0)}\n")
        f.write(f"- ASHA rung budgets (epochs): {budgets}\n")

        completed = [t for t in trials if t.get("best_val_mpjpe_mm") is not None]
        if completed:
            best = min(completed, key=lambda t: t["best_val_mpjpe_mm"])
            f.write("\n## Best trial\n\n")
            f.write(f"- Trial ID: {best['trial_id']}\n")
            f.write(f"- Best val MPJPE: {best['best_val_mpjpe_mm']:.2f} mm\n")
            f.write(f"- Slug: `{best['slug']}`\n")
            f.write(f"- Checkpoint: `{best['output']}`\n")

        f.write("\n## All trial records\n\n")
        f.write(
            "| ID | d | rh | nst | lr | pp | epi | vis | unc | vdo | mv | bs | ts | "
            "rung | epochs | ret | best_mm | elapsed_min |\n"
        )
        f.write(
            "|----|---|----|-----|----|----|-----|-----|-----|-----|----|----|----|"
            "------|------|-----|---------|-------------|\n"
        )
        for t in trials:
            cfg = t["config"]
            best = t.get("best_val_mpjpe_mm")
            best_str = f"{best:.2f}" if best is not None else "n/a"
            elapsed_min = "n/a" if t.get("elapsed_sec") is None else f"{t['elapsed_sec']/60:.1f}"
            f.write(
                f"| {t['trial_id']} | {cfg['d']} | {cfg['residual_hidden']} | "
                f"{cfg['n_st_layers']} | {cfg['lr']:.0e} | {cfg['pp_loss_weight']} | "
                f"{cfg['epipolar_loss_weight']} | {cfg['visibility_loss_weight']} | "
                f"{cfg['uncertainty_loss_weight']} | {cfg['view_dropout_rate']} | "
                f"{cfg['min_views']} | {cfg['batch_size']} | {cfg['train_samples']} | "
                f"{t.get('rung', 'n/a')} | {t.get('epochs', 'n/a')} | "
                f"{t.get('returncode', 'n/a')} | {best_str} | {elapsed_min} |\n"
            )
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.mode == "smoke":
        space = smoke_search_space(omniview=args.omniview)
        n_trials = max(1, args.n_trials)
        max_epochs = min(args.epochs, 2)
        device = "cpu"
    else:
        space = default_search_space(omniview=args.omniview)
        n_trials = args.n_trials
        max_epochs = args.epochs
        device = args.device

    # Smoke mode forces CPU and smoke data regardless of --full.
    args.full = False if args.mode == "smoke" else args.full
    budgets = asha_rung_epochs(max_epochs, max(1, args.asha_rungs))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = generate_trials(space, args.mode, n_trials, args.seed)

    trials_json_path = Path(args.resume) if args.resume else output_dir / "trials.json"
    results: List[Dict[str, Any]] = []
    completed_keys: set = set()
    if trials_json_path.exists():
        with open(trials_json_path) as f:
            results = json.load(f)
        for r in results:
            completed_keys.add((r["trial_id"], r["rung"]))
        print(f"Resumed from {trials_json_path} ({len(results)} previous records)")

    # For dry-run smoke tests, just print a few commands and exit cleanly.
    if args.dry_run and args.mode == "smoke":
        print("\n=== Dry-run smoke: commands that would be executed ===\n")
        for trial in trials:
            output_path = output_dir / f"{trial.slug()}.pth"
            cmd = build_command(trial, args, budgets[-1], output_path, dry_run=True)
            print(" ".join(cmd))
        return

    try:
        for trial in trials:
            # ASHA: start from the first rung and promote survivors.
            for rung_idx, rung_epochs in enumerate(budgets):
                key = (trial.trial_id, rung_idx)
                if key in completed_keys:
                    print(f"  Skipping already-completed ({trial.trial_id}, rung {rung_idx})")
                    continue

                output_path = output_dir / f"{trial.slug()}_rung{rung_idx}.pth"
                if args.skip_existing and output_path.exists():
                    print(f"  Skipping existing checkpoint {output_path}")
                    continue

                print(
                    f"\n=== Trial {trial.trial_id:03d}/{len(trials):03d} "
                    f"rung {rung_idx + 1}/{len(budgets)} ({rung_epochs} epochs): {trial.slug()} ==="
                )

                cmd = build_command(trial, args, rung_epochs, output_path, dry_run=args.dry_run)

                record: Dict[str, Any] = {
                    "trial_id": trial.trial_id,
                    "rung": rung_idx,
                    "epochs": rung_epochs,
                    "slug": trial.slug(),
                    "config": trial.to_dict(),
                    "output": str(output_path),
                }

                result = run_trial(cmd, dry_run=args.dry_run, device=device)
                record.update(result)
                results.append(record)

                # Save after every rung.
                with open(trials_json_path, "w") as f:
                    json.dump(results, f, indent=2)

                if result["returncode"] != 0:
                    print(
                        f"  Trial {trial.trial_id} rung {rung_idx} failed "
                        f"with return code {result['returncode']}"
                    )
                    break  # Do not promote failed trials.

                print(
                    f"  Trial {trial.trial_id} rung {rung_idx} done: "
                    f"best_val_mpjpe_mm={result.get('best_val_mpjpe_mm')}, "
                    f"elapsed={(result.get('elapsed_sec') or 0)/60:.1f} min"
                )

                # Successive halving: only promote top fraction to next rung.
                if rung_idx < len(budgets) - 1:
                    current_rung_records = [
                        r for r in results if r["rung"] == rung_idx and r.get("best_val_mpjpe_mm") is not None
                    ]
                    if current_rung_records:
                        sorted_records = sorted(
                            current_rung_records,
                            key=lambda r: r["best_val_mpjpe_mm"],
                        )
                        cutoff = max(1, len(sorted_records) // 2)
                        promoted_ids = {r["trial_id"] for r in sorted_records[:cutoff]}
                        if trial.trial_id not in promoted_ids:
                            print(
                                f"  Trial {trial.trial_id} not promoted beyond rung {rung_idx}"
                            )
                            break
    finally:
        write_report(results, output_dir, budgets)
        print(f"\nReport written to {output_dir / 'hp_search_report.md'}")
        print(f"Trials JSON: {trials_json_path}")


if __name__ == "__main__":
    main()
