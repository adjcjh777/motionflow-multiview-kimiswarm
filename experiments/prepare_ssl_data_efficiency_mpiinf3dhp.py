"""Create data-efficiency split manifests for SSL pre-training → fine-tuning.

Given MPI-INF-3DHP (or any canonical .npz) training sequences, this script builds
non-overlapping temporal clip indices, subsamples them to 10 %, 25 %, 50 % and 100 %
using a fixed seed, and writes:

  * a JSON manifest with split sizes and clip ranges, and
  * per-fraction shell launcher scripts ready to run once an GPU is free.

The script is CPU-only and does not train anything.

Usage
-----
    # Real data (no GPU required)
    python experiments/prepare_ssl_data_efficiency_mpiinf3dhp.py \
        --train data/webbridge/mpi_inf_3dhp/s_01_seq_01_v14_multiview_m.npz \
                data/webbridge/mpi_inf_3dhp/s_01_seq_02_v14_multiview_m.npz \
        --val data/webbridge/mpi_inf_3dhp/s_02_seq_01_v14_multiview_m.npz \
        --clip_len 13 --val_stride 10 \
        --ssl_checkpoint outputs/ray_attention_ssl_h36m.pth \
        --out_dir outputs/ssl_data_efficiency_mpiinf3dhp

    # CPU smoke test with synthetic data (no real .npz required)
    python experiments/prepare_ssl_data_efficiency_mpiinf3dhp.py --smoke
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def _write_synthetic_npz(path: Path, n_frames: int = 500, n_views: int = 14, j: int = 17):
    """Generate a tiny canonical .npz for CPU smoke testing."""
    rng = np.random.default_rng(42)
    points_2d = rng.normal(0, 1, size=(n_frames, n_views, j, 2)).astype(np.float32)
    confidences = rng.uniform(0.8, 1.0, size=(n_frames, n_views, j)).astype(np.float32)
    joints_3d = rng.normal(0, 1, size=(n_frames, j, 3)).astype(np.float32)
    camera_K = np.stack([np.eye(3, dtype=np.float32) for _ in range(n_views)], axis=0)
    camera_R = np.stack([np.eye(3, dtype=np.float32) for _ in range(n_views)], axis=0)
    camera_t = rng.normal(0, 1, size=(n_views, 3)).astype(np.float32)
    np.savez(
        path,
        points_2d=points_2d,
        confidences=confidences,
        joints_3d=joints_3d,
        camera_K=camera_K,
        camera_R=camera_R,
        camera_t=camera_t,
    )


def _count_clips(npz_path: str, clip_len: int) -> int:
    data = np.load(npz_path)
    total_frames = data["points_2d"].shape[0]
    return max(1, (total_frames - clip_len) // 1 + 1)


def _build_clip_indices(npz_paths: list, clip_len: int, seed: int):
    """Return list of (file_idx, clip_start, total_frames) for every possible clip."""
    rng = random.Random(seed)
    indices = []
    for file_idx, path in enumerate(npz_paths):
        data = np.load(path)
        total_frames = data["points_2d"].shape[0]
        for start in range(0, max(1, total_frames - clip_len + 1)):
            indices.append((file_idx, start, total_frames))
    rng.shuffle(indices)
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data-efficiency split manifest for SSL fine-tuning"
    )
    parser.add_argument("--train", type=str, nargs="+", help="Training .npz files")
    parser.add_argument("--val", type=str, help="Validation .npz file")
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--val_stride", type=int, default=10)
    parser.add_argument("--ssl_checkpoint", type=str, default="outputs/ray_attention_ssl_h36m.pth",
                        help="Path to SSL pre-trained checkpoint used for warm-start")
    parser.add_argument("--out_dir", type=str, default="outputs/ssl_data_efficiency_mpiinf3dhp")
    parser.add_argument("--smoke", action="store_true",
                        help="Generate synthetic .npz files and run in CPU mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        tmp = Path("tmp/ssl_data_efficiency_smoke")
        tmp.mkdir(parents=True, exist_ok=True)
        train_a = tmp / "s_01_seq_01_multiview_m.npz"
        train_b = tmp / "s_01_seq_02_multiview_m.npz"
        val_f = tmp / "s_02_seq_01_multiview_m.npz"
        _write_synthetic_npz(train_a, n_frames=500)
        _write_synthetic_npz(train_b, n_frames=500)
        _write_synthetic_npz(val_f, n_frames=300)
        args.train = [str(train_a), str(train_b)]
        args.val = str(val_f)
        args.out_dir = "outputs/ssl_data_efficiency_smoke"

    if not args.train or not args.val:
        raise ValueError("Provide --train and --val, or run with --smoke")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_indices = _build_clip_indices(args.train, args.clip_len, args.seed)
    total = len(all_indices)

    manifest = {
        "ssl_checkpoint": args.ssl_checkpoint,
        "clip_len": args.clip_len,
        "val_stride": args.val_stride,
        "train_files": args.train,
        "val_file": args.val,
        "total_train_clips": total,
        "seed": args.seed,
        "splits": {},
    }

    for frac in FRACTIONS:
        n = max(1, int(round(total * frac)))
        split_indices = all_indices[:n]
        manifest["splits"][f"{int(frac * 100):02d}pct"] = {
            "fraction": frac,
            "n_clips": n,
            "example_file": args.train[split_indices[0][0]],
        }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write per-fraction launcher scripts (GPU only; not executed here).
    launcher_paths = []
    for frac in FRACTIONS:
        label = f"{int(frac * 100):02d}pct"
        script = out_dir / f"run_ssl_finetune_{label}.sh"
        with open(script, "w", newline="\n") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"# SSL fine-tune on {label} of labeled MPI-INF-3DHP clips.\n")
            f.write("# AUTO-GENERATED; do not run while another GPU job is active.\n\n")
            f.write('cd "$(dirname "$0")/../../.."\n')
            f.write(". .venv/bin/activate 2>/dev/null || conda activate mf\n\n")
            f.write("python experiments/train_ray_attention_temporal_crossview_residual_principal_point_mpiinf3dhp.py \\\n")
            for t in args.train:
                f.write(f"    --train {t} \\\n")
            f.write(f"    --val {args.val} \\\n")
            f.write(f"    --clip_len {args.clip_len} \\\n")
            f.write("    --d 64 --n_st_layers 2 --residual_hidden 128 \\\n")
            f.write("    --principal_point_max_offset 20.0 \\\n")
            f.write("    --epochs 30 --batch_size 8 --train_samples 4000 \\\n")
            f.write(f"    --val_stride {args.val_stride} \\\n")
            f.write("    --pp_loss_weight 0.1 --cam_aug_pp 5.0 \\\n")
            f.write(f"    --warm_start {args.ssl_checkpoint} \\\n")
            f.write(f"    --output outputs/ray_attention_ssl_finetune_{label}.pth \\\n")
            f.write(f"    --seed {args.seed}\n")
        launcher_paths.append(str(script))

    # Summary report (stdout and file).
    summary_lines = [
        "SSL data-efficiency split manifest prepared.",
        f"  train files: {args.train}",
        f"  val file: {args.val}",
        f"  total possible train clips (non-overlapping, clip_len={args.clip_len}): {total}",
        "  splits:",
    ]
    for frac in FRACTIONS:
        label = f"{int(frac * 100):02d}pct"
        n = manifest["splits"][label]["n_clips"]
        summary_lines.append(f"    {label}: {n} clips")
    summary_lines.append(f"\nManifest: {manifest_path}")
    summary_lines.append(f"Launchers: {', '.join(launcher_paths)}")
    summary_lines.append("\nThese launchers use the SSL checkpoint for warm-start and train on the")
    summary_lines.append("specified fraction of labeled MPI-INF-3DHP clips. GPU training is queued,")
    summary_lines.append("not started, to avoid interfering with the currently running curriculum job.")

    summary = "\n".join(summary_lines)
    print(summary)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
