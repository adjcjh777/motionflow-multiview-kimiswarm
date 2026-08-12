"""Batch-convert the preprocessed Hugging Face Human3.6M archive to canonical .npz.

Each output file covers one (subject, action, split) group in the canonical format used
by the ray-attention training pipeline.

.. warning::
    Without true 3D mocap ground truth, the produced labels are circular
    (DLT triangulation of the input 2D keypoints). The converter now raises
    an error unless true-GT ``.npz`` files are found in ``data/h36m_true_gt/``
    or ``--allow-circular-fallback`` is explicitly passed.

Example
-------
    conda run -n mf python experiments/batch_convert_h36m_webbridge.py \
        --data_root data/h36m_hf \
        --out_dir data/webbridge/h36m \
        --splits train test
"""

import argparse
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.webbridge_loader import convert_human36m


def discover_groups(archive_path: Path):
    """Return (subjects, splits) present in the archive."""
    with zipfile.ZipFile(archive_path) as z:
        with z.open("h36m_sh_conf_cam_source_final.pkl") as f:
            data = pickle.load(f)

    subject_action_split = set()
    for split in data.keys():
        for src in data[split]["source"]:
            m = re.match(r"s_(\d+)_act_(\d+).*", src)
            if m:
                subject = int(m.group(1))
                action = int(m.group(2))
                subject_action_split.add((subject, action, split))
    return subject_action_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/h36m_hf")
    parser.add_argument("--out_dir", type=str, default="data/webbridge/h36m")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        choices=["train", "test"])
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="If provided, only convert these subjects.")
    parser.add_argument("--actions", type=int, nargs="+", default=None,
                        help="If provided, only convert these actions.")
    parser.add_argument("--meters", action="store_true",
                        help="Also emit a meters variant (_m.npz).")
    parser.add_argument("--allow-circular-fallback", action="store_true",
                        help="Allow triangulation fallback when true 3D GT is missing.")
    parser.add_argument("--true-gt-dir", type=Path, default=None,
                        help="Directory to search for true-GT npz files (default: data/h36m_true_gt).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    archive_path = data_root / "h36m_sh_conf_cam_source_final.pkl.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    all_groups = discover_groups(archive_path)
    if args.subjects:
        all_groups = {(s, a, sp) for (s, a, sp) in all_groups if s in args.subjects}
    if args.actions:
        all_groups = {(s, a, sp) for (s, a, sp) in all_groups if a in args.actions}
    if not all_groups:
        print("No groups to convert.")
        return

    # Group by split for nicer logging
    by_split = defaultdict(list)
    for s, a, sp in sorted(all_groups):
        by_split[sp].append((s, a))

    for split in args.splits:
        groups = by_split.get(split, [])
        if not groups:
            continue
        print(f"Converting {len(groups)} (subject, action) groups for split={split}")
        for subject, action in groups:
            out_path = out_dir / f"s_{subject:02d}_acts_{action:02d}_multiview.npz"
            if out_path.exists():
                print(f"  Skipping existing {out_path}")
                continue
            try:
                convert_human36m(
                    data_root=data_root,
                    subject=subject,
                    actions=[action],
                    split=split,
                    out_dir=out_dir,
                    archive_file=archive_path.name,
                    true_gt_dir=args.true_gt_dir,
                    allow_circular_fallback=args.allow_circular_fallback,
                )
                print(f"  -> {out_path}")
                if args.meters:
                    # Load and emit meters variant
                    npz = np.load(out_path)
                    np.savez(
                        str(out_path).replace(".npz", "_m.npz"),
                        points_2d=npz["points_2d"],
                        confidences=npz["confidences"],
                        joints_3d=npz["joints_3d"] / 1000.0,
                        camera_K=npz["camera_K"],
                        camera_R=npz["camera_R"],
                        camera_t=npz["camera_t"] / 1000.0,
                    )
            except Exception as e:
                print(f"  Failed s{subject:02d}a{action:02d} ({split}): {e}")


if __name__ == "__main__":
    main()
