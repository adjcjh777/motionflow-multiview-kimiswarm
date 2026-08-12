#!/usr/bin/env python3
"""Build corrected, non-circular H36M canonical .npz files from true mocap 3D GT.

This converter replaces the legacy circular labels in ``data/h36m_hf/``
(where ``joints_3d`` was a DLT triangulation of the input 2D) with the
official mocap world coordinates from ``data/h36m_true_gt/data_3d_h36m.npz``.
The stored 2D keypoints come from the preprocessed Hugging Face archive, so
the 3D/2D/camera triplet is physically consistent up to the detector noise
(~16 mm direct MJE on a clean re-triangulation audit).

The output is written in **meters** (the convention used by current training
configs) to a new directory so existing files are not overwritten.

Example
-------
    python scripts/convert_h36m_true_gt_v2.py \
        --subject 1 --actions 2 --split train \
        --out_dir data/h36m_true_gt_v2

Output
------
    data/h36m_true_gt_v2/s_01_act_02_multiview.npz

Verification
------------
    python scripts/diagnose_circular_labels.py \
        data/h36m_true_gt_v2/s_01_act_02_multiview.npz

A reasonable result is direct MJE in the tens of millimetres (detector noise),
and definitely not 0 mm (circular DLT labels) nor thousands of mm (unit/coordinate
misalignment).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from motionflow_mv.data.webbridge_loader import convert_human36m
from experiments.prepare_h36m_true_gt import build_true_gt_from_data3d_npz


MM_PER_M = 1000.0


def _build_true_gt_npz(
    data3d_npz: Path,
    pkl_path: Path,
    split: str,
    subject: int,
    actions: list[int],
) -> Path:
    """Build a temporary true-GT npz containing ``joints_3d`` in pkl frame order."""
    joints_3d_mm = build_true_gt_from_data3d_npz(
        data3d_npz=data3d_npz,
        pkl_path=pkl_path,
        split=split,
        subject=subject,
        actions=actions,
    )
    temp = tempfile.NamedTemporaryFile(suffix="_true_gt.npz", delete=False)
    np.savez_compressed(temp.name, joints_3d=joints_3d_mm)
    temp.close()
    return Path(temp.name)


def convert_h36m_true_gt_v2(
    subject: int,
    actions: list[int],
    split: str,
    out_dir: Path,
    data3d_npz: Path,
    pkl_path: Path,
    data_root: Path,
) -> Path:
    """Convert one H36M subject/action/split to a true-GT canonical .npz in meters.

    Parameters
    ----------
    subject: subject id, e.g. ``1``.
    actions: list of action ids, e.g. ``[2, 3]``.
    split: ``"train"`` or ``"test"``.
    out_dir: directory where the output .npz will be written.
    data3d_npz: path to the VideoPose3D-format ``data_3d_h36m.npz``.
    pkl_path: path to the preprocessed ``h36m_sh_conf_cam_source_final.pkl.zip``.
    data_root: directory containing ``camera_params.json``.

    Returns
    -------
    Path to the generated canonical .npz (in meters).
    """
    true_gt_npz = _build_true_gt_npz(
        data3d_npz=data3d_npz,
        pkl_path=pkl_path,
        split=split,
        subject=subject,
        actions=actions,
    )
    try:
        mm_path = convert_human36m(
            data_root=data_root,
            subject=subject,
            actions=actions,
            split=split,
            out_dir=out_dir / "_tmp_mm",
            true_gt_path=true_gt_npz,
        )
    finally:
        true_gt_npz.unlink(missing_ok=True)

    # Convert the mm canonical npz to meters (current project convention).
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / mm_path.name
    with np.load(mm_path) as data:
        np.savez(
            final_path,
            points_2d=data["points_2d"],
            confidences=data["confidences"],
            joints_3d=data["joints_3d"] / MM_PER_M,
            camera_K=data["camera_K"],
            camera_R=data["camera_R"],
            camera_t=data["camera_t"] / MM_PER_M,
        )

    # Clean up temporary mm file.
    mm_path.unlink(missing_ok=True)
    try:
        (out_dir / "_tmp_mm").rmdir()
    except OSError:
        pass

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert H36M to a true-mocap canonical .npz in meters."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject id.")
    parser.add_argument(
        "--actions", type=int, nargs="+", required=True, help="Action ids."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"]
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/h36m_true_gt_v2"),
        help="Output directory.",
    )
    parser.add_argument(
        "--data3d-npz",
        type=Path,
        default=Path("data/h36m_true_gt/data_3d_h36m.npz"),
        help="Path to the official data_3d_h36m.npz release.",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path("data/h36m_hf/h36m_sh_conf_cam_source_final.pkl.zip"),
        help="Path to the preprocessed H36M pkl zip archive.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/h36m_hf"),
        help="Directory containing camera_params.json.",
    )
    args = parser.parse_args()

    out_path = convert_h36m_true_gt_v2(
        subject=args.subject,
        actions=args.actions,
        split=args.split,
        out_dir=args.out_dir,
        data3d_npz=args.data3d_npz,
        pkl_path=args.pkl,
        data_root=args.data_root,
    )
    print(f"Saved true-GT v2 canonical npz to {out_path}")


if __name__ == "__main__":
    main()
