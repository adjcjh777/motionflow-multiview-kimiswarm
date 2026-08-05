"""Run a training script repeatedly with different seeds and log a manifest.

This helper makes it easy to produce 3–5 checkpoints for statistical reporting.
It is model-agnostic: it simply calls the provided training script with
``--seed <seed>`` and ``--output <out_dir>/<base_name>_seed<seed>.pth``.

Example
-------
    python experiments/run_repeated_seeds.py \
        --script experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
        --base_args "--train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz --clip_len 13 --d 64 --residual_hidden 128 --n_st_layers 2 --epochs 20 --batch_size 8 --train_samples 4000 --pp_loss_weight 0.05" \
        --seeds 42 43 44 \
        --out_dir outputs/repeated_seeds \
        --base_name crossview_residual_pp
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run training with multiple seeds and produce a manifest.")
    parser.add_argument("--script", type=str, required=True, help="Path to the training script to run")
    parser.add_argument("--base_args", type=str, required=True, help="Arguments to pass to the training script (quoted)")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of random seeds")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store per-seed checkpoints")
    parser.add_argument("--base_name", type=str, required=True, help="Base name for output checkpoints")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "script": args.script,
        "base_args": args.base_args,
        "seeds": args.seeds,
        "checkpoints": {},
    }

    for seed in args.seeds:
        checkpoint = out_dir / f"{args.base_name}_seed{seed}.pth"
        command = [
            sys.executable,
            args.script,
            *args.base_args.split(),
            "--seed", str(seed),
            "--output", str(checkpoint),
        ]
        print(f"\n=== Seed {seed} ===")
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"[ERROR] Training failed for seed {seed}")
                manifest["checkpoints"][seed] = {"path": str(checkpoint), "status": "failed"}
            else:
                manifest["checkpoints"][seed] = {"path": str(checkpoint), "status": "completed"}
        else:
            manifest["checkpoints"][seed] = {"path": str(checkpoint), "status": "dry_run"}

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
