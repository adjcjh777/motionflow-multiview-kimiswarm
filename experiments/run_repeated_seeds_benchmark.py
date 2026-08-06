"""Multi-seed benchmark launcher built on :class:`BenchmarkProtocol`.

This script is the thin command-line interface around
:class:`motionflow_mv.eval.BenchmarkProtocol`.  It runs a training script with
multiple seeds, writes one manifest per seed, and aggregates them into a top-level
``manifest.json``.

Example
-------
    # Dry-run: only print commands and write manifests (no GPU)
    python experiments/run_repeated_seeds_benchmark.py \
        --script experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
        --base_args "--train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz --epochs 20" \
        --seeds 42 43 44 \
        --out_dir outputs/repeated_seeds/pp_model \
        --base_name crossview_residual_pp \
        --dry_run

    # Real run (GPU, queued)
    python experiments/run_repeated_seeds_benchmark.py \
        --script experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \
        --base_args "--train ... --val ... --epochs 20" \
        --seeds 42 43 44 45 46 \
        --out_dir outputs/repeated_seeds/pp_model \
        --base_name crossview_residual_pp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.eval import BenchmarkConfig, BenchmarkProtocol


def main():
    parser = argparse.ArgumentParser(
        description="Run a training script over multiple seeds and write per-seed manifests."
    )
    parser.add_argument("--script", type=str, required=True, help="Training script path")
    parser.add_argument("--base_args", type=str, required=True, help="Quoted base arguments")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of seeds")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base_name", type=str, required=True, help="Checkpoint name prefix")
    parser.add_argument("--dataset", type=str, default="mpiinf3dhp", help="Dataset name for config")
    parser.add_argument("--split", type=str, default="test", help="Split name for config")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate manifests without launching training",
    )
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        seed=args.seeds[0],
    )
    protocol = BenchmarkProtocol(cfg)

    manifest = protocol.run_multi_seed(
        script=args.script,
        base_args=args.base_args,
        seeds=args.seeds,
        out_dir=args.out_dir,
        base_name=args.base_name,
        dry_run=args.dry_run,
    )

    print(json.dumps(manifest, indent=2))
    print(f"\nAggregate manifest: {Path(args.out_dir) / 'manifest.json'}")
    print("Per-seed manifests written to:", Path(args.out_dir))


if __name__ == "__main__":
    main()
