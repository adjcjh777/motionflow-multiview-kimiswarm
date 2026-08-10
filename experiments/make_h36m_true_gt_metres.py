"""Convert the H36M true-GT canonical npz files (mm) to metre convention.

The project's training manifests use metre-convention ``*_m.npz`` files
(``joints_3d`` and ``camera_t`` in metres, ``camera_K`` unchanged), matching
``data/webbridge/h36m_meters/``. The freshly regenerated true-GT canonical
npz under ``data/h36m_true_gt/`` are in millimetres (camera_t norm ~5500).

This script divides ``joints_3d`` and ``camera_t`` by 1000 and writes
``<stem>_m.npz`` next to each input. 2D keypoints, confidences and camera_K
are untouched.

Usage:
    python experiments/make_h36m_true_gt_metres.py \
        --glob "data/h36m_true_gt/s_*_multiview.npz"

Author: research swarm (data foundation repair, 2026-08)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def convert(src: Path, dst: Path) -> None:
    data = np.load(src)
    out = {}
    for k in data.files:
        arr = data[k]
        if k in ("joints_3d", "camera_t"):
            arr = arr.astype(np.float64) / 1000.0
        out[k] = arr
    np.savez_compressed(dst, **out)
    t_norm = float(np.linalg.norm(out["camera_t"][0]))
    print(
        f"{src.name} -> {dst.name}: joints_3d {out['joints_3d'].shape}, "
        f"range [{out['joints_3d'].min():.3f}, {out['joints_3d'].max():.3f}] m, "
        f"camera_t norm {t_norm:.3f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="data/h36m_true_gt/s_*_multiview.npz",
        help="glob of mm canonical npz to convert",
    )
    args = parser.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob}")
    for f in files:
        src = Path(f)
        dst = src.with_name(src.stem + "_m.npz")
        if dst.exists():
            print(f"skip (exists): {dst.name}")
            continue
        convert(src, dst)


if __name__ == "__main__":
    main()
