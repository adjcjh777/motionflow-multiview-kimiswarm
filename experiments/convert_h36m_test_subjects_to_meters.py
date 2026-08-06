#!/usr/bin/env python3
"""Convert Human3.6M test-subject .npz files from millimeters to meters.

The canonical WebBridge H36M files under ``data/webbridge/h36m`` are stored in
millimeters (camera_t ~ 5e3, joints_3d ~ 1e3--2e4).  Evaluation expects meters.
This script converts all action files for the standard test subjects S9 and S11
and writes ``data/webbridge/h36m_meters/s_09/11_acts_XX_multiview_m.npz``.

It is CPU-only, read-only on the source files, and safe to run while GPU
jobs are training.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def convert_file(input_path: Path, output_path: Path, scale: float = 1000.0) -> None:
    data = dict(np.load(input_path))
    required = {"points_2d", "confidences", "joints_3d", "camera_K", "camera_R", "camera_t"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{input_path} missing canonical keys: {missing}")

    data["camera_t"] = data["camera_t"] / scale
    data["joints_3d"] = data["joints_3d"] / scale
    # points_2d (pixels), K, R unchanged

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert H36M S9/S11 .npz files to meters.")
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("data/webbridge/h36m"),
        help="Source directory containing s_09/s_11 .npz files in mm.")
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path("data/webbridge/h36m_meters"),
        help="Output directory for meter-scale .npz files.")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Division factor to convert source units to meters.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned conversions without writing files.")
    args = parser.parse_args(argv)

    subjects = ["s_09", "s_11"]
    src_paths = []
    for subj in subjects:
        src_paths.extend(sorted(args.src_root.glob(f"{subj}_acts_*.npz")))

    if not src_paths:
        print(f"No H36M S9/S11 .npz files found under {args.src_root}", file=sys.stderr)
        return 1

    print(f"Found {len(src_paths)} files to convert")
    for src in src_paths:
        dst = args.dst_root / f"{src.stem}_m.npz"
        print(f"  {src.name} -> {dst}")
        if not args.dry_run:
            convert_file(src, dst, args.scale)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
