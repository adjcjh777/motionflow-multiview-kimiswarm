"""Convert AIST++ annotations to canonical WebBridge ``.npz`` files.

Usage
-----
    # Convert every sequence (after running scripts/download_aistpp.py)
    conda run -n mf python experiments/convert_aistpp_v1.py \
        --data_root data/webbridge/aistpp \
        --out data/webbridge/aistpp_canonical

    # Convert only the first 5 sequences (smoke test)
    conda run -n mf python experiments/convert_aistpp_v1.py \
        --data_root data/webbridge/aistpp \
        --out data/webbridge/aistpp_canonical \
        --max_seqs 5

The script writes one ``.npz`` per sequence using the canonical keys:
``points_2d``, ``confidences``, ``joints_3d``, ``camera_K``, ``camera_R``,
``camera_t``. By default ``joints_3d`` and ``camera_t`` are multiplied by
``0.01`` because AIST++ raw units are centimeters; disable with ``--raw``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.webbridge_loader import convert_aistpp


def main():
    parser = argparse.ArgumentParser(description="Convert AIST++ to canonical .npz.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path to extracted AIST++ annotations.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the canonical .npz files.",
    )
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=None,
        help="Convert at most N sequences (useful for smoke tests).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Keep raw AIST++ units instead of scaling to meters.",
    )
    args = parser.parse_args()

    out_paths = convert_aistpp(
        data_root=args.data_root,
        out_dir=args.out,
        scale_factor=None if args.raw else 0.01,
        max_seqs=args.max_seqs,
    )

    print(f"Converted {len(out_paths)} sequences.")
    if out_paths:
        sample = np.load(out_paths[0])
        print("Sample:", out_paths[0].name)
        print("  points_2d  ", sample["points_2d"].shape)
        print("  confidences", sample["confidences"].shape)
        print("  joints_3d  ", sample["joints_3d"].shape)
        print("  camera_K   ", sample["camera_K"].shape)


if __name__ == "__main__":
    main()
