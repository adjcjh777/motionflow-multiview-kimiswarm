"""Orchestrator for the publishable-benchmark experimental plan (swarm_iter6).

The script defines a ranked list of experiments, derives the exact commands,
and can optionally run a short smoke-test subset on the local RTX 4090.

Usage
-----
    # Print the plan without running anything
    conda run -n mf python experiments/run_publishable_benchmark_plan.py

    # Run the smoke-test subset (<=10 epochs, <=30 min)
    conda run -n mf python experiments/run_publishable_benchmark_plan.py --run --smoke

    # Run a slightly longer fast pass (still bounded)
    conda run -n mf python experiments/run_publishable_benchmark_plan.py --run --fast
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent

MPI_DIR = ROOT / "data" / "webbridge" / "mpi_inf_3dhp"
H36M_DIR = ROOT / "data" / "h36m_hf"
SHELF_DIR = ROOT / "data" / "shelf_campus"


@dataclass
class Experiment:
    name: str
    priority: int
    purpose: str
    train_cmd: List[str] = field(default_factory=list)
    eval_cmd: List[str] = field(default_factory=list)
    dataset: str = ""
    metric_goal: str = ""  # human-readable target / comparison
    compute: str = "low"  # low / medium / high
    depends_on: List[str] = field(default_factory=list)


MPI_TRAIN = [
    str(MPI_DIR / "s_01_seq_01_v14_multiview_m.npz"),
    str(MPI_DIR / "s_01_seq_02_v14_multiview_m.npz"),
]
MPI_VAL = str(MPI_DIR / "s_02_seq_01_v14_multiview_m.npz")
MPI_VAL_SEQ2 = str(MPI_DIR / "s_02_seq_02_v14_multiview_m.npz")
MPI_TEST = str(MPI_DIR / "s_03_seq_01_v14_multiview_m.npz")

H36M_TRAIN = str(H36M_DIR / "s_01_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz")
H36M_VALS = {
    "s_05": str(H36M_DIR / "s_05_acts_02_multiview.npz"),
    "s_09": str(H36M_DIR / "s_09_acts_02_multiview.npz"),
    "s_11": str(H36M_DIR / "s_11_acts_02_multiview.npz"),
}

SHELF_VAL = str(SHELF_DIR / "Shelf_Seq1" / "pseudogt_m.npz")
CAMPUS_VAL = str(SHELF_DIR / "Campus_Seq1" / "pseudogt_m.npz")


def base_train(output: str, train_files: List[str], val_file: str, *,
               clip_len: int = 13, d: int = 64, residual_hidden: int = 128,
               epochs: int = 1, batch_size: int = 2, train_samples: int = 1000,
               extra: List[str] = None) -> List[str]:
    cmd = [
        "python", "experiments/train_ray_attention_temporal_residual_v3_mpiinf3dhp.py",
        "--train", *train_files,
        "--val", val_file,
        "--clip_len", str(clip_len),
        "--d", str(d),
        "--residual_hidden", str(residual_hidden),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--train_samples", str(train_samples),
        "--output", output,
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def base_eval(checkpoint: str, val_file: str, *, clip_len: int = 13,
              d: int = 64, residual_hidden: int = 128, out: str) -> List[str]:
    return [
        "python", "experiments/eval_ray_attention_temporal_residual_v1.py",
        "--checkpoint", checkpoint,
        "--val", val_file,
        "--clip_len", str(clip_len),
        "--d", str(d),
        "--residual_hidden", str(residual_hidden),
        "--out", out,
    ]


def _maybe_smoke_mpi(paths: List[str], smoke: bool) -> List[str]:
    """If smoke mode and the 200-frame smoke subsets exist, use them."""
    if not smoke:
        return paths

    def _smoke_counterpart(p: str) -> str:
        # Map full MPI path to its 200-frame smoke subset in tmp/.
        name = Path(p).stem
        if "s_01_seq_01" in name:
            return str(ROOT / "tmp" / "mpi_s01_seq01_smoke.npz")
        if "s_01_seq_02" in name:
            return str(ROOT / "tmp" / "mpi_s01_seq02_smoke.npz")
        if "s_02_seq_01" in name:
            return str(ROOT / "tmp" / "mpi_s02_seq01_smoke.npz")
        return p

    smoke_paths = [_smoke_counterpart(p) for p in paths]
    if all(Path(p).exists() for p in smoke_paths):
        return smoke_paths
    return paths


def build_plan(smoke: bool = True, fast: bool = False) -> List[Experiment]:
    """Return the swarm_iter6 experimental plan.

    Priorities are 1 (highest) to 5 (lowest).  Smoke runs shorten every
    experiment to a bounded training epoch / metric-only evaluation.  If the
    200-frame MPI smoke subsets exist, they are used in smoke mode for speed.
    """
    # Choose paths: smoke subsets when available, otherwise full datasets.
    mpi_train = _maybe_smoke_mpi(MPI_TRAIN, smoke)
    mpi_val = str(_maybe_smoke_mpi([MPI_VAL], smoke)[0])

    plan = []
    epochs = 1 if smoke else (3 if fast else 10)
    train_samples = 100 if smoke else (4000 if fast else 8000)
    batch_size = 2 if smoke else 4
    clip_len = 13

    # ------------------------------------------------------------------
    # Priority 1: Scale the residual model on MPI-INF-3DHP (core track)
    # ------------------------------------------------------------------
    plan.append(Experiment(
        name="residual_mpi_d128_h256_clip13",
        priority=1,
        purpose="Scale residual model: d=128, residual_hidden=256, clip_len=13 on MPI-INF-3DHP S1->S2.",
        train_cmd=base_train(
            "outputs/swarm_iter6_residual_mpi_d128_h256_clip13.pth",
            mpi_train, mpi_val,
            clip_len=clip_len, d=128, residual_hidden=256,
            epochs=epochs, batch_size=batch_size, train_samples=train_samples,
            extra=["--scheduler", "cosine", "--log", "outputs/swarm_iter6_residual_mpi_d128_h256_clip13_log.json"],
        ),
        eval_cmd=base_eval(
            "outputs/swarm_iter6_residual_mpi_d128_h256_clip13.pth",
            mpi_val,
            clip_len=clip_len, d=128, residual_hidden=256,
            out="outputs/swarm_iter6_residual_mpi_d128_h256_clip13_eval.json",
        ),
        dataset="MPI-INF-3DHP (train S1 seq1/seq2, val S2 seq1)",
        metric_goal="Beat current best 13.84 mm MPJPE on S2 Seq1.",
        compute="medium",
    ))

    plan.append(Experiment(
        name="residual_mpi_d128_h256_clip27",
        priority=1,
        purpose="Longer temporal context: d=128, residual_hidden=256, clip_len=27.",
        train_cmd=base_train(
            "outputs/swarm_iter6_residual_mpi_d128_h256_clip27.pth",
            mpi_train, mpi_val,
            clip_len=27, d=128, residual_hidden=256,
            epochs=epochs, batch_size=2, train_samples=train_samples,
            extra=["--scheduler", "cosine", "--log", "outputs/swarm_iter6_residual_mpi_d128_h256_clip27_log.json"],
        ),
        eval_cmd=base_eval(
            "outputs/swarm_iter6_residual_mpi_d128_h256_clip27.pth",
            mpi_val,
            clip_len=27, d=128, residual_hidden=256,
            out="outputs/swarm_iter6_residual_mpi_d128_h256_clip27_eval.json",
        ),
        dataset="MPI-INF-3DHP (train S1, val S2 seq1)",
        metric_goal="Improve temporal consistency and PCK/AUC; monitor memory (clip_len=27 doubles memory).",
        compute="high",
        depends_on=["residual_mpi_d128_h256_clip13"],
    ))

    # ------------------------------------------------------------------
    # Priority 2: Cross-subject H36M benchmark
    # ------------------------------------------------------------------
    plan.append(Experiment(
        name="residual_h36m_s01_train_s05_val",
        priority=2,
        purpose="Train residual model on H36M S01 actions and evaluate cross-subject on S05.",
        train_cmd=base_train(
            "outputs/swarm_iter6_residual_h36m_s01.pth",
            [H36M_TRAIN], H36M_VALS["s_05"],
            clip_len=clip_len, d=128, residual_hidden=256,
            epochs=epochs, batch_size=batch_size, train_samples=train_samples,
            extra=["--scheduler", "cosine", "--log", "outputs/swarm_iter6_residual_h36m_s01_log.json"],
        ),
        eval_cmd=base_eval(
            "outputs/swarm_iter6_residual_h36m_s01.pth",
            H36M_VALS["s_05"],
            clip_len=clip_len, d=128, residual_hidden=256,
            out="outputs/swarm_iter6_residual_h36m_s05_eval.json",
        ),
        dataset="Human3.6M (train S01, val S05)",
        metric_goal="Compare to VoxelPose / DLT baseline on H36M cross-subject.",
        compute="medium",
    ))

    # ------------------------------------------------------------------
    # Priority 3: Auxiliary losses + robustness curriculum
    # ------------------------------------------------------------------
    plan.append(Experiment(
        name="residual_mpi_aux_bone_velocity",
        priority=3,
        purpose="Add bone-length and velocity-consistency auxiliary losses on top of residual model.",
        train_cmd=base_train(
            "outputs/swarm_iter6_residual_mpi_aux.pth",
            mpi_train, mpi_val,
            clip_len=clip_len, d=128, residual_hidden=256,
            epochs=epochs, batch_size=batch_size, train_samples=train_samples,
            extra=[
                "--aux_weight", "0.001",
                "--velocity_weight", "0.001",
                "--scheduler", "cosine",
                "--log", "outputs/swarm_iter6_residual_mpi_aux_log.json",
            ],
        ),
        eval_cmd=base_eval(
            "outputs/swarm_iter6_residual_mpi_aux.pth",
            mpi_val,
            clip_len=clip_len, d=128, residual_hidden=256,
            out="outputs/swarm_iter6_residual_mpi_aux_eval.json",
        ),
        dataset="MPI-INF-3DHP",
        metric_goal="Improve temporal coherence and joint-length plausibility; expect modest MPJPE gain.",
        compute="medium",
        depends_on=["residual_mpi_d128_h256_clip13"],
    ))

    # ------------------------------------------------------------------
    # Priority 4: Cross-scene zero-shot on Shelf / Campus
    # ------------------------------------------------------------------
    # These experiments do not train; they re-use the MPI checkpoint.
    for val_name, val_path, d_val in [
        ("shelf", SHELF_VAL, "Shelf"),
        ("campus", CAMPUS_VAL, "Campus"),
    ]:
        plan.append(Experiment(
            name=f"residual_mpi_zs_{val_name}",
            priority=4,
            purpose=f"Zero-shot evaluation of MPI-trained residual model on {d_val}.",
            train_cmd=[],
            eval_cmd=base_eval(
                "outputs/swarm_iter6_residual_mpi_d128_h256_clip13.pth",
                val_path,
                clip_len=clip_len, d=128, residual_hidden=256,
                out=f"outputs/swarm_iter6_residual_mpi_zs_{val_name}_eval.json",
            ),
            dataset=d_val,
            metric_goal="Report cross-scene generalisation; expect higher error than in-dataset.",
            compute="low",
            depends_on=["residual_mpi_d128_h256_clip13"],
        ))

    # ------------------------------------------------------------------
    # Priority 5: Strong baselines for publication table
    # ------------------------------------------------------------------
    plan.append(Experiment(
        name="small_residual_baseline_mpi",
        priority=5,
        purpose="Small residual model (d=64, residual_hidden=128) as a compact baseline on MPI.",
        train_cmd=base_train(
            "outputs/swarm_iter6_small_residual_baseline.pth",
            mpi_train, mpi_val,
            clip_len=clip_len, d=64, residual_hidden=128,
            epochs=epochs, batch_size=batch_size, train_samples=train_samples,
        ),
        eval_cmd=base_eval(
            "outputs/swarm_iter6_small_residual_baseline.pth",
            mpi_val,
            clip_len=clip_len, d=64, residual_hidden=128,
            out="outputs/swarm_iter6_small_residual_baseline_eval.json",
        ),
        dataset="MPI-INF-3DHP",
        metric_goal="Compact baseline; the scaled residual model should substantially outperform this.",
        compute="low",
    ))

    return plan


def run_command(cmd: List[str], env: dict = None) -> subprocess.CompletedProcess:
    """Run a command in the project root and return its result."""
    env = env or os.environ.copy()
    return subprocess.run(cmd, cwd=str(ROOT), env=env, check=False, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description="Publishable benchmark plan (swarm_iter6)")
    parser.add_argument("--run", action="store_true", help="Actually execute the plan subset")
    parser.add_argument("--smoke", action="store_true", help="Use smoke-test hyperparameters (1 epoch, small samples)")
    parser.add_argument("--fast", action="store_true", help="Use fast-pass hyperparameters (3 epochs)")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiment names to run (default: all)")
    parser.add_argument("--out", type=str, default="outputs/swarm_iter6_plan.json",
                        help="Where to write the JSON plan/result summary")
    args = parser.parse_args()

    plan = build_plan(smoke=args.smoke, fast=args.fast)

    # Filter experiments if requested.
    if args.experiments:
        names = set(args.experiments.split(","))
        plan = [e for e in plan if e.name in names]

    # Print plan.
    print("=" * 70)
    print("SWARM_ITER6 PUBLISHABLE BENCHMARK PLAN")
    print("=" * 70)
    for e in plan:
        print(f"\n[{e.priority}] {e.name}")
        print(f"    Purpose: {e.purpose}")
        print(f"    Dataset: {e.dataset}")
        print(f"    Metric goal: {e.metric_goal}")
        print(f"    Compute: {e.compute}")
        if e.depends_on:
            print(f"    Depends on: {', '.join(e.depends_on)}")
        if e.train_cmd:
            print(f"    Train: {' '.join(e.train_cmd)}")
        if e.eval_cmd:
            print(f"    Eval:  {' '.join(e.eval_cmd)}")

    results = []
    if args.run:
        print("\n" + "=" * 70)
        print("RUNNING SMOKE TESTS")
        print("=" * 70)
        for e in plan:
            print(f"\n--- {e.name} ---")
            result = {
                "name": e.name,
                "priority": e.priority,
                "dataset": e.dataset,
            }
            if e.train_cmd:
                print(f"Training: {' '.join(e.train_cmd)}")
                train_res = run_command(e.train_cmd)
                result["train_returncode"] = train_res.returncode
                result["train_stdout_tail"] = "\n".join(train_res.stdout.splitlines()[-20:])
                if train_res.returncode != 0:
                    result["train_stderr_tail"] = "\n".join(train_res.stderr.splitlines()[-20:])
                    print("TRAIN FAILED:", train_res.returncode)
                    print(train_res.stderr[-500:])
                else:
                    print("TRAIN OK")
                    print(train_res.stdout[-500:])
            if e.eval_cmd:
                print(f"Evaluating: {' '.join(e.eval_cmd)}")
                eval_res = run_command(e.eval_cmd)
                result["eval_returncode"] = eval_res.returncode
                result["eval_stdout_tail"] = "\n".join(eval_res.stdout.splitlines()[-20:])
                if eval_res.returncode != 0:
                    result["eval_stderr_tail"] = "\n".join(eval_res.stderr.splitlines()[-20:])
                    print("EVAL FAILED:", eval_res.returncode)
                    print(eval_res.stderr[-500:])
                else:
                    print("EVAL OK")
                    print(eval_res.stdout[-500:])
            results.append(result)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"experiments": results}, f, indent=2, default=str)
        print(f"\nSaved run summary to {out_path}")
    else:
        print("\nNo --run given; printed plan only.")


if __name__ == "__main__":
    main()
